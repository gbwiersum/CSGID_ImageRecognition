#! usr/bin/env python3
from PIL import Image

def make_prediction(model, imagepath):
    #image = Image(imagepath)
    #image = image.to_numpy()
    pred = model.predict(image)
    return pred