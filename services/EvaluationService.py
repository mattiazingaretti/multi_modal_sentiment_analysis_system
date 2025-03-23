

from sklearn.metrics import classification_report
import torch


class EvaluationService:
    def __init__(self, model, val_loader,data_loader, device):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.dataloader = data_loader

    def evaluate_model(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.dataloader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, target_names=['Neutral', 'Positive', 'Negative'])
        return all_labels, all_preds, report