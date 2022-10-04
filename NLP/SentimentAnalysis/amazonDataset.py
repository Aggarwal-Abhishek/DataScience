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

class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        
        sentimentMap = {'__label__1': 0, '__label__2': 1}
        self.lines = []
        self.labels = []
        
        with open(file_path) as fi:
            for line in fi.readlines():
                line = line.split(' ')
                label = line[0]
                
                line = ' '.join(line[1:])
                if ': ' in line:
                    line = line.split(': ')
                    self.lines.append(line[0])
                    self.lines.append(' '.join(line[1:]))
                    self.labels.append(sentimentMap[label])
                    self.labels.append(sentimentMap[label])
                else:
                    self.lines.append(line)
                    self.labels.append(sentimentMap[label])

    def __getitem__(self, idx):
        text = self.lines[idx]
        target = self.labels[idx]
        
        text = Clean(text)
        return text, target
    
    def __len__(self):
        return len(self.lines)
 
