import cv2
from torch.utils.data import Dataset
import os
from torchvision import transforms


class SentimentImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = []
        self.labels = []

        label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
        for label_name, label in label_map.items():
            folder_path = os.path.join(data_dir, label_name)
            for img_name in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        if self.transform:
            image = self.transform(image)

        return image, label