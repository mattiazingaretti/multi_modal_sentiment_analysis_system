from sklearn.model_selection import train_test_split
import torch
from constants import Constants
from models.SentimentClassifier import SentimentClassifier
from models.SentimentImageDataset import SentimentImageDataset
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from constants import Constants
from models.SentimentImageDataset import SentimentImageDataset
import torch.nn as nn
from torchvision import models
import torch.optim as optim

class ImageClassifierBuilder:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Output shape: (batch_size, 2048, 1, 1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()

        self.classifier = SentimentClassifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

    
    def load_images(self):
        self.dataset = SentimentImageDataset(Constants.IMAGES_DIR)
        train_data, val_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=32, shuffle=False)



    def train_model(self, verbose=True, num_epochs=10):
        for epoch in range(num_epochs):
            self.classifier.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    features = self.model(images).squeeze()

                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

    def evaluate_model(self):
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                features = self.model(images).squeeze()

                outputs = self.classifier(features)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")


    def build_classifier(self):
        self.load_images()
        self.train_model()
        self.evaluate_model()
