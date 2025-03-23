# multi_modal_sentiment_analysis_system

## Objective
The goal of this project is to create a system that performs sentiment analysis on a dataset that includes both textual and image data. You will need to:

Preprocess both types of data.
Build models to analyze sentiment in text (using NLP techniques).
Analyze sentiment in images (using computer vision techniques).
Combine both models to produce a final sentiment classification (positive, negative, neutral).

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


### Key Metrics Analysis (from last training_metrics.txt commit)


#### Understanding the Metrics

**Accuracy Metrics:**
- **Precision:** Measures how many of the predicted positives are actually correct
  - Formula: True Positives / (True Positives + False Positives)
  - Our model: 0.96-0.99 across all classes

- **Recall:** Measures how many of the actual positives were identified correctly
  - Formula: True Positives / (True Positives + False Negatives)
  - Our model: 0.94-0.99 across all classes

- **F1-Score:** Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Our model: 0.96-0.97 across all classes

**Loss Metrics:**
- **Training Loss:** Measures model's error on training data
  - Lower values indicate better learning
  - Our model: Decreased from 1.08 to 0.45

- **Validation Loss:** Measures model's error on unseen data
  - Used to monitor overfitting
  - Our model: Stabilized around 0.29

**Support:** Number of samples for each class in the test set
- Neutral: 1404 samples
- Positive: 1323 samples
- Negative: 1168 samples

These metrics indicate a well-balanced, highly accurate model with consistent performance across all sentiment classes.

#### Training Progress Analysis
**Overall Performance Improvement:**
- **Initial Accuracy:** 60% (Epoch 1)
- **Final Accuracy:** 97% (Epoch 20)
- Significant improvement in all metrics over the training period.

**Loss Trends:**
- **Training Loss:** Decreased from 1.0820 to 0.4506
- **Validation Loss:** Stabilized around 0.2900
- Consistent decrease in training loss indicates effective learning.

#### Class-wise Performance (Final Epoch)
**Key Observations:**
- The model achieved balanced performance across all classes.
- Major improvements occurred between epochs 10-15.
- Best validation performance at epoch 20 with a 0.97 F1-score.
- No signs of overfitting; training and validation metrics remain stable.

#### Learning Phases
- **Early Phase (Epochs 1-5):** Rapid improvement from 60% to 83% accuracy.
- **Middle Phase (Epochs 6-15):** Steady improvement to 93% accuracy.
- **Late Phase (Epochs 16-20):** Fine-tuning to reach 97% accuracy.

### Some Examples from the Gradio Interface:

![alt text](image.png)

![alt text](image-1.png)

