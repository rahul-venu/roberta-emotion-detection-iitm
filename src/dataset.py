import torch
from torch.utils.data import Dataset
from config import *

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128, has_labels=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            labels = self.df.loc[self.df.index[idx], LABELS].values.astype(float)
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item