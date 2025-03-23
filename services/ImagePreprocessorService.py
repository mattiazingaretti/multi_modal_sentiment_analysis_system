from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.MultiModalDataset import MultiModalDataset
from torchvision import transforms
from constants import Constants

class ImagePreprocessorService:
    def __init__(self, text_dataset):
        self.text_dataset = text_dataset
        self.image_dataset = MultiModalDataset(self.text_dataset, img_dir=Constants.IMAGES_DIR, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        train_data, val_data = train_test_split(self.image_dataset, test_size=0.2, random_state=42)

        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    