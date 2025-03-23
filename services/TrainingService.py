
import torch
from constants import Constants
from services.EvaluationService import EvaluationService
from services.ImagePreprocessorService import ImagePreprocessorService
from services.ModelInitializerService import ModelInitializerService


class TrainingService:

    def __init__(self, modelInitializerService : ModelInitializerService, imagePreprocessorService: ImagePreprocessorService):
        self.model = modelInitializerService.model
        self.val_loader = imagePreprocessorService.val_loader
        self.device = modelInitializerService.device
        self.criterion = modelInitializerService.criterion
        self.optimizer = modelInitializerService.optimizer
        self.scheduler = modelInitializerService.scheduler
        self.train_loader = imagePreprocessorService.train_loader
    
    def _train_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        return running_loss / len(dataloader)

    
    def train(self, num_epochs = 20, best_val_loss = float('inf')):
        
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(self.model, self.train_loader, self.criterion, self.optimizer)
            val_labels, val_preds , report = EvaluationService(self.model, self.val_loader,self.train_loader, self.device).evaluate_model()
            
            val_preds_tensor = torch.tensor(val_preds, dtype=torch.float32).to(self.device)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).to(self.device)
        
            val_loss = self.criterion(
                val_preds_tensor,
                val_labels_tensor
            ).item()
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Constants.BEST_MODEL_PATH)
            
            with open('training_metrics.txt', 'a') as f:
                f.write(f'Epoch {epoch+1}/{num_epochs}\n')
                f.write(report)
                f.write(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n')