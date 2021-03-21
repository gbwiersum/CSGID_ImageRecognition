from datetime import datetime
import keras
import mlflow
from PIL import Image
import numpy as np
import pandas as pd
import glob


class CrystalFinder:
    classifier = None
    version = ""

    def __init__(self, classifier_path):
        self.classifier = keras.models.load_model(classifier_path)
        self.version = classifier_path

    @staticmethod
    def image_preprocess(image_path):
        image = Image.open(image_path)
        crop_size = min(image.size)
        left = (image.size[1] - crop_size) // 2
        right = image.size[1] - (image.size[1] - crop_size) // 2
        bottom = (image.size[0] - crop_size) // 2
        top = image.size[0] // 2 + crop_size // 2
        image = image.crop((left, bottom, right, top))
        image = image.resize((299, 299), Image.ANTIALIAS)
        image = np.asarray(image).reshape((1, 299, 299, 3))
        return image

    def image_getter(self, plate_dir):
        wells = pd.DataFrame(glob.glob(plate_dir + "/*"), columns=["filepath"])
        wells = wells.reset_index().drop(columns="index")
        wells["image"] = wells['filepath'].apply(lambda x: self.image_preprocess(x))
        return wells

    def get_predictions(self, plate_folder):
        wells = self.image_getter(plate_folder)
        wells["AI_predict"] = wells['image'].apply(lambda x: self.score(x))
        return wells

    def score(self, image):
        # for each image, load, resize and predict. (morphological?)
        prediction = np.argmax(self.classifier.predict(image), axis=1)[0]
        if prediction == 0:
            return "Clear"
        if prediction == 1:
            return "Crystal"
        if prediction == 2:
            return "Precipitate"
        if prediction == 3:
            return "Other"

    @staticmethod
    def get_human_scored():
        try:
            filepath = input("Path to human-scored csv: ")
            updates = pd.read_csv(filepath)
            return updates
        except():
            print("An error was encountered, please try again")

    def update_model(self):
        updates = self.get_human_scored()
        mlflow.keras.autolog()
        self.classifier.fit(updates, batch_size=10, epochs=100, steps=1000)
        # TODO: MLFlow?
        version = str(datetime.now().date())
        self.classifier.save("../Models/" + version + "-ImageModel.h5")
        print("A new classifier was saved at: " + str(version))
