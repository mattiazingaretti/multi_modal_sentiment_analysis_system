# multi_modal_sentiment_analysis_system

## Objective
The goal of this project is to create a system that performs sentiment analysis on a dataset that includes both textual and image data. You will need to:

Preprocess both types of data.
Build models to analyze sentiment in text (using NLP techniques).
Analyze sentiment in images (using computer vision techniques).
Combine both models to produce a final sentiment classification (positive, negative, neutral).
## Tasks
1. Dataset Preparation:
Use a publicly available multi-modal dataset, such as the Twitter Sentiment AnalysisLinks to an external site. with Images dataset, which includes both text and corresponding image data.
The dataset should include:
    - Text data: Tweets, captions, or any textual content.
    - Image data: Corresponding images, memes, or attached photos.
    - Sentiment labels: Each data entry should have a sentiment label—positive, negative, or neutral.

2. NLP Component (Text Analysis):
Preprocess the text data by applying standard techniques such as tokenization, stopword removal, and lowercasing.
Extract features from the text using popular word embeddings such as Word2Vec, GloVe, or BERT embeddings.
Build a sentiment classification model for the text data using an NLP framework like Hugging Face, Scikit-learn, or TensorFlow.

3. Computer Vision Component (Image Analysis):
Preprocess the image data by resizing and normalizing the images for model input.
Use a pre-trained Convolutional Neural Network (CNN) such as ResNet or VGG to extract features from the images.
Build a sentiment classification model for the image data.

4. Fusion and Final Classification:
Combine the features from both the text and image models. You can use a concatenation layer or any other fusion technique to merge the feature vectors.
Train a final model (such as a fully connected layer) to perform sentiment classification based on the combined features from both the text and image models.

## Data Used citation

[1] D. J. Mohammed and H. J. Aleqabie, "The Enrichment Of MVSA Twitter Data Via Caption-Generated Label Using Sentiment Analysis," 2022 Iraqi International Conference on Communication and Information Technologies (IICCIT), Basrah, Iraq, 2022, pp. 322-327, doi: 10.1109/IICCIT55816.2022.10010435.

The Original Dataset (Which We Developed on):

[2] T. Niu, S. A. Zhu, L. Pang and A. El Saddik, Sentiment Analysis on Multi-view Social Data, MultiMedia Modeling (MMM), pp: 15-27, Miami, 2016.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi_modal_sentiment_analysis_system.git
cd multi_modal_sentiment_analysis_system
```

2. Create and activate a virtual environment:
```bash
  python -m venv venv .\venv\Scripts\activate
```

3. Install dependencies:
```bash
  pip install -r requirements.txt
```

4. Place your images (I used the fancy dataset provided) in a 'data' directory in the current project with this structure:



## How to Use

### Run the server (with venv active for current logged user)
```bash
    python main.py
```

### Training the Model
1. Start the training process through the API endppoint:
```bash
 curl -X POST http://localhost:8080/train_model
```

### Testing the Model
You can test the model in three ways:

1. Using the Gradio Interface:
   - Open http://localhost:4200 in your browser after you launched the server and they will be up and running.
   - Upload an image and enter text
   - Click "Submit" to get the outcome

2. Using the FastAPI endpoint:
```bash
curl -X POST http://localhost:8080/test_model \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This is some fancy text",
         "image_path": "path/to/your/image.jpg"
     }'
```

3. Using Python code directly in your existing projects:
```python
from pipelines.PipeLineBuilder import PipeLineBuilder


pipeline = PipeLineBuilder()
results = pipeline.test_model(
    text="This is a happy image",
    model_path="best_multimodal_model.pth",
    image_path="path/to/your/image.jpg"
)
```

## Model Architecture

The system uses a multi-modal architecture combining vision and text models:

### Text Processing
- BERT base uncased for text feature extraction
- Input: Raw text
- Output: 768-dimensional text embedding

### Image Processing
- ResNet50 pretrained on ImageNet
- Input: 224x224 RGB images
- Output: 2048-dimensional image embedding

### Fusion Architecture
```
Text Input → BERT → 768d
                        → Concatenate → FC(2816→1024) → ReLU → 
Image Input → ResNet → 2048d        → Dropout → FC(1024→512) → ReLU →
                                    → Dropout → FC(512→3) → Softmax
```

Key Features:
- Multi-modal fusion through feature concatenation
- Dropout layers (0.5) for regularization
- Batch normalization for stable training
- Three-class output (Positive, Negative, Neutral)
- Cross-entropy loss function
- Adam optimizer with learning rate scheduler


