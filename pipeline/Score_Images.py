import keras
import os
from keras.models import  load_model
import pandas as pd
from datetime import date
from ImageRecognition.pipeline.Image_getter import Image_getter
from ImageRecognition.pipeline.make_prediction import make_prediction


def score_images():
  #Load model with highest index value
  model_list = os.walk('../SavedModels')
  model = load_model(model_list[-1][-1])

  df = Image_getter()
  df["AIScore"] = df["ImagePath"].apply(lambda x: make_prediction(model, x))

  #TODO: Write predictions to csv with image indexes
  df.to_csv("../predictions/"+str(date)+"_predictions")

  #TODO: generate performance report

  #TODO: frontend for re-scoring

  #TODO: retrain on human-scoring
