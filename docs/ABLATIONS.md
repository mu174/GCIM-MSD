# Ablation implementation and evidence

| Variant | Implemented operation |
|---|---|
| w/o-CRMD | Replace the CRMD decoder slot with zeros. |
| w/o-MCM | Set the MCM loss coefficient to zero. |
| w/o-Gate | Replace task sigmoid gates with multipliers of one. |
| w/o-SP | Replace the shared additional-modality projection with independent modality projections. The global graph projection is retained. |
| w/o T / w/o V | Retrain with the omitted modality excluded from the rebuilt global graph and CRMD/MCM inputs. The legacy loader still requires its cache. |
| STL | Report sarcasm from stl_sarcasm and sentiment from stl_sentiment; each checkpoint is selected by validation F1 of its supervised task. |

The variants use `AblationModel`. Full-model, ablation, and modality-removal
results are provided in the summary files under `results/`.
