from services.TextPreprocessorService import TextPreprocessorService


class PipeLineBuilder:
    
    def __init__(self):
        pass

    def build_image_classifier_pipeline(self):
        textPreprocessor = TextPreprocessorService()
        text_dataset = textPreprocessor.produce()
        
