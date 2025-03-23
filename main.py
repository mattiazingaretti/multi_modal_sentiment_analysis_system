import threading
from constants import Constants
from models.dto.TestModelRequestDTO import TestModelRequestDTO
from models.dto.TestModelResponseDTO import TestModelResponseDTO
from pipelines.PipeLineBuilder import PipeLineBuilder
from fastapi import FastAPI
import uvicorn
import gradio as gr

app = FastAPI()
pipeline = PipeLineBuilder()

@app.post("/train_model")
def train_model():
  print("Starting Training model")
  pipeline.train_model()
  return {"message": "Model training started"}

@app.post("/test_model")
def test_model(request: TestModelRequestDTO):
  results = pipeline.test_model(request.text, Constants.BEST_MODEL_PATH,request.image_path)
  resp = TestModelResponseDTO(results['prediction'], results['confidence'], results['probabilities']) 
  return {"message": "Model Succefully tests", "prediction": resp.prediction, "confidence": resp.confidence, "probabilities": resp.proability_breakdown}

def gradio_test_model(text, image):
  request = TestModelRequestDTO(text=text, image_path=image.name)
  response = test_model(request)
  return response


# there is no place like home :)
def run_gradio(interface: gr.Interface):
  interface.launch(share=True, server_name="127.0.0.1", server_port=4200)

def run_fastapi():
  uvicorn.run(app, host=Constants.SERVER_HOST_NAME, port=Constants.SERVER_PORT)


if __name__ == "__main__":
  interface = gr.Interface(
    fn=gradio_test_model,
    inputs=["text", "file"],
    outputs="json",
    title="Multi-Modal Sentiment Analysis",
    description="Upload an image and enter text to test the sentiment analysis model."
  )

  gradio_thread = threading.Thread(target=run_gradio, args=(interface,))
  fastapi_thread = threading.Thread(target=run_fastapi)

  gradio_thread.start()
  fastapi_thread.start()

  gradio_thread.join()
  fastapi_thread.join() 
