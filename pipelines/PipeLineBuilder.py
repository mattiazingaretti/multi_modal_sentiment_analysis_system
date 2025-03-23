from services.ModelInitializerService import ModelInitializerService
from services.TestModelService import TestModelService
from services.TextPreprocessorService import TextPreprocessorService
from services.ImagePreprocessorService import ImagePreprocessorService
from services.TrainingService import TrainingService

class PipeLineBuilder:
    
    def __init__(self):
        self.text_dataset = TextPreprocessorService().produce()
        self.model_init = ModelInitializerService()
        self.image_preprocessor = ImagePreprocessorService(self.text_dataset)
        self.text_preprocessor = TextPreprocessorService()
    
    def train_model(self):
        TrainingService(self.model_init, self.image_preprocessor).train()

    def test_model(self,  text, model_path, image_path):
        return TestModelService(self.model_init, self.text_preprocessor).test( text, model_path, image_path)