import torch
from torchvision import transforms
from transformers import BertTokenizer
import cv2
from services import TextPreprocessorService
from services.ModelInitializerService import ModelInitializerService


class TestModelService:
    def __init__(self, modelInitializer : ModelInitializerService, textPreprocessor : TextPreprocessorService):
        self.device = modelInitializer.device
        self.model = modelInitializer.model
        self.textPreprocessor = textPreprocessor

    def test(self, text, model_path, image_path):
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoding = tokenizer.encode_plus(
            self.textPreprocessor.preprocess_text(text),
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            raise ValueError("Could not read image. Check the file path and format.")
            
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor, input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)

        label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
        result = {
            'prediction': label_map[preds.item()],
            'confidence': confidence.item(),
            'probabilities': {label_map[i]: prob.item() 
                            for i, prob in enumerate(probabilities[0])}
        }
        print(f"Prediction: {result['prediction']} ({result['confidence']:.2%} confidence)")
        print("Probability breakdown:")
        for label, prob in result['probabilities'].items():
            print(f"- {label}: {prob:.2%}")
        return result