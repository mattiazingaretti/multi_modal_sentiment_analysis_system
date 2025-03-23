import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
import cv2
import os


class MultiModalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['LABEL']
        text = row['cleaned_text']
        img_path = os.path.join(self.img_dir, row['file_path'])
        
        if not img_path.endswith(".jpg"):
            raise ValueError("Image file must be in .jpg format.")
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Could not read image. Check the file path and format")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }