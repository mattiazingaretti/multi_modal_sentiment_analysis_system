import kagglehub
from kagglehub import KaggleDatasetAdapter

from pipelines.ImageClassifierBuilder import ImageClassifierBuilder


tweets_path = "LabeledText.xlsx"

tweets_df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "dunyajasim/twitter-dataset-for-sentiment-analysis",
  tweets_path
)



if __name__ == "__main__":
  imgClassifier = ImageClassifierBuilder()
  imgClassifier.build_classifier()
