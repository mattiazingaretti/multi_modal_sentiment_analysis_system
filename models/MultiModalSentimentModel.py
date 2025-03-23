import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.image_model = models.resnet50(pretrained=True)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
        for param in self.image_model.parameters():
            param.requires_grad = False
            
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_model(image)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten to [batch_size, features]

        text_outputs = self.text_model(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] # [batch_size, 768]
        
        combined = torch.cat([img_features, text_features], dim=1)
        
        x = self.dropout1(torch.relu(self.bn1(self.fc1(combined))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x