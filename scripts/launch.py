#!/usr/bin/env python3
"""Read experiment settings, snapshot code, and run isolated jobs serially."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import yaml

VARIANTS = ('wo_crmd', 'wo_mcm', 'wo_gate', 'wo_sp', 'wo_t', 'wo_v',
            'stl_sarcasm', 'stl_sentiment')
ROOT = Path(__file__).resolve().parents[1]


def load_config(path, dataset):
    c = yaml.safe_load(path.read_text())
    expected = {'dataset', 'seed', 'epochs', 'batch_size', 'learning_rate',
                'gradient_accumulation_steps', 'loss', 'checkpoint_selection',
                'test_evaluation'}
    if set(c) != expected:
        raise ValueError('Unexpected or missing configuration keys')
    if c['dataset'].lower() != dataset:
        raise ValueError('Configuration dataset mismatch')
    loss = c['loss']
    if set(loss) != {'lambda_sarcasm', 'lambda_sentiment', 'lambda_mcm',
                      'mcm_margin', 'sentiment_weighting'}:
        raise ValueError('Unexpected loss settings')
    s = c['checkpoint_selection']
    if set(s) != {'split', 'metric', 'sarcasm_weight', 'sentiment_weight', 'tie_breaker'}:
        raise ValueError('Unexpected checkpoint selection settings')
    if (s['split'], s['metric'], s['tie_breaker']) != (
            'validation', 'weighted_f1', 'sarcasm_weighted_f1'):
        raise ValueError('Only validation weighted F1 selection is implemented')
    if c['test_evaluation'] != 'once_after_loading_best_checkpoint':
        raise ValueError('Unsupported test evaluation policy')
    if min(s['sarcasm_weight'], s['sentiment_weight']) < 0 or abs(
            s['sarcasm_weight'] + s['sentiment_weight'] - 1) > 1e-12:
        raise ValueError('Selection weights must be nonnegative and sum to one')
    for name in ('epochs', 'batch_size', 'gradient_accumulation_steps'):
        if not isinstance(c[name], int) or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(c['seed'], int) or c['learning_rate'] <= 0:
        raise ValueError('Invalid seed or learning rate')
    if loss['sentiment_weighting'] not in ('none', 'balanced'):
        raise ValueError('Unsupported sentiment weighting')
    if any(loss[k] < 0 for k in ('lambda_sarcasm', 'lambda_sentiment', 'lambda_mcm', 'mcm_margin')):
        raise ValueError('Loss weights and margin must be nonnegative')
    return c


def command(c, dataset, variant, source, checkpoint):
    entry = 'train_ablation.py' if variant != 'full' else f'train_{dataset}.py'
    cmd = [sys.executable, '-u', str(source / entry)]
    if variant != 'full':
        cmd += ['--dataset', dataset, '--variant', variant]
    for key in ('seed', 'epochs', 'batch_size', 'learning_rate', 'gradient_accumulation_steps'):
        cmd += ['--' + key.replace('_', '-'), str(c[key])]
    for key, value in c['loss'].items():
        cmd += ['--' + key.replace('_', '-'), str(value)]
    for task in ('sarcasm', 'sentiment'):
        cmd += [f'--selection-{task}-weight', str(c['checkpoint_selection'][task+'_weight'])]
    return cmd + ['--checkpoint', str(checkpoint)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('mustard', 'memotion'))
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--runs-root', type=Path, default=ROOT/'runs')
    p.add_argument('--config', type=Path)
    p.add_argument('--variants', nargs='+', choices=('full',)+VARIANTS, default=['full'])
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if len(set(a.variants)) != len(a.variants):
        p.error('Variants must be unique')
    config_path = a.config or ROOT/'configs'/f'{a.dataset}.yaml'
    c = load_config(config_path, a.dataset)
    if a.plan_only:
        print(json.dumps({'config': c, 'variants': a.variants, 'device': a.device,
                          'command_example': command(c, a.dataset, a.variants[0],
                          Path('source_snapshot'), Path('artifacts/best_model.pth'))}, indent=2))
        return
    if not a.data_root.is_dir():
        p.error('Data root does not exist')
    run = a.runs_root.expanduser().resolve() / (
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+a.dataset+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    source = run/'source_snapshot'
    source.mkdir()
    for f in (ROOT/'src').glob('*.py'):
        shutil.copy2(f, source/f.name)
    shutil.copy2(__file__, run/'launch.py')
    (run/'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    commands = {v: command(c, a.dataset, v, source, run/v/'artifacts/best_model.pth')
                for v in a.variants}
    env = os.environ.copy()
    env.update(GCIM_MSD_DATA_ROOT=str(a.data_root.expanduser().resolve()),
               PYTHONHASHSEED=str(c['seed']), PYTHONUNBUFFERED='1',
               PYTHONPATH=str(source), GCIM_MSD_DEVICE=a.device)
    if a.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    packages = {}
    for name in ('torch', 'dgl', 'dgl-cu113', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    spec = {'created_at': datetime.now(timezone.utc).isoformat(), 'config': c,
            'commands': commands, 'python': sys.version, 'packages': packages,
            'environment': {k: env.get(k) for k in ('GCIM_MSD_DATA_ROOT', 'PYTHONHASHSEED',
                         'CUDA_VISIBLE_DEVICES', 'GCIM_MSD_DEVICE')},
            'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in source.glob('*.py')}}
    (run/'RUN_SPEC.json').write_text(json.dumps(spec, indent=2))
    status = {v: {'state': 'pending'} for v in a.variants}
    def save_status():
        temporary = run/'STATUS.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(run/'STATUS.json')
    save_status()
    print('Archive:', run, flush=True)
    for variant, cmd in commands.items():
        job = run/variant
        job.mkdir(exist_ok=False)
        status[variant] = {'state': 'running'}
        save_status()
        try:
            with (job/'train.log').open('x') as log:
                with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = proc.wait()
            status[variant] = {'state': 'completed' if code == 0 else 'failed', 'return_code': code}
            save_status()
            if code:
                raise SystemExit(code)
        except (Exception, KeyboardInterrupt) as exc:
            status[variant] = {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                               'error': type(exc).__name__}
            save_status()
            raise


if __name__ == '__main__':
    main()
