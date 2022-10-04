import re
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

import torch
CLEAN_REGEX = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
# stemmer = SnowballStemmer("english")

def Clean(text):
    text = re.sub(CLEAN_REGEX, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            # tokens.append(stemmer.stem(token))
            tokens.append(token)
    return " ".join(tokens).strip()

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
    
    def __getitem__(self, idx):
        target, text = self.df.iloc[idx]
        
        text = Clean(text)
        if int(target) == 4:
            target = 1
        else:
            target = 0
        
        return text, target
    
    def __len__(self):
        return len(self.df)

def ReadDF(file_path, train_ratio=False, return_dataset=True):
    df = pd.read_csv(file_path, encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])
    df.drop(columns=['ids', 'date', 'flag', 'user'], inplace=True)
    
    if train_ratio:
        train_len = int(len(df) * train_ratio)
        idxs = np.random.permutation(len(df))
        
        train_df = df.iloc[idxs[:train_len]]
        test_df = df.iloc[idxs[train_len:]]
        
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        if return_dataset:
            return TwitterDataset(train_df), TwitterDataset(test_df)
        return train_df, test_df
    
    if return_dataset:
        return TwitterDataset(df)
    return df
