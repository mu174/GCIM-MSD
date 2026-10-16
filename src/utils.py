# Importing libraries
import random 
import numpy as np
import pandas as pd
import torch
import config, dataset
import csv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = []
            uttDict[name]['sentiment-label'] = []
            uttDict[name]['emotion-label'] = []
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'].append(item[4])
            uttDict[item[0]]['sentiment-label'] .append(item[5])
            uttDict[item[0]]['emotion-label'].append(item[7])
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


def set_up_memotion():
    """Load the fixed Memotion train, test, and validation graph datasets."""
    
    df_train = pd.read_csv(f'{config.root_dir}{config.me_train_csv_name}', encoding='latin')
    # df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.me_test_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)

    df_dev = pd.read_csv(f'{config.root_dir}{config.me_dev_csv_name}', encoding='latin')
    # df_test = df_test.dropna().reset_index(drop=True)

    # print("Training DataFrame columns:", df_train.columns)
    # print("Testing DataFrame columns:", df_test.columns)
    # print("Development DataFrame columns:", df_dev.columns)
    
    dataset_train = dataset.GraphDataset_Memotion(df_train, config.root_dir, config.me_image_vec_dir, config.me_text_vec_dir)
    
    dataset_test = dataset.GraphDataset_Memotion(df_test, config.root_dir, config.me_image_vec_dir, config.me_text_vec_dir)

    dataset_dev = dataset.GraphDataset_Memotion(df_dev, config.root_dir, config.me_image_vec_dir, config.me_text_vec_dir)
    
    return dataset_train, dataset_test, dataset_dev


def set_up_mustard():
    """Load the fixed MUStARD train, test, and validation graph datasets."""

    dataset_train = dataset.GraphDataset_Mustard('train', config.root_dir, config.mu_image_vec_dir, config.mu_text_vec_dir, config.mu_audio_vec_dir)

    dataset_test = dataset.GraphDataset_Mustard('test', config.root_dir, config.mu_image_vec_dir, config.mu_text_vec_dir, config.mu_audio_vec_dir)

    dataset_dev = dataset.GraphDataset_Mustard('dev', config.root_dir, config.mu_image_vec_dir, config.mu_text_vec_dir, config.mu_audio_vec_dir)

    return dataset_train, dataset_test, dataset_dev
