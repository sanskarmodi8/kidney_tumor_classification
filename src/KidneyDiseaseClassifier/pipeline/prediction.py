import numpy as np
import tensorflow as tf
import os
from KidneyDiseaseClassifier.config.configuration import ConfigurationManager
from pathlib import Path

image = tf.keras.preprocessing.image
load_model = tf.keras.models.load_model

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        
        # load model
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model = load_model(Path(training_config.trained_model_path))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Cyst'
            return [prediction}]
        elif result[0] == 1:
            prediction = 'Normal'
            return [prediction}]
        elif result[0] == 2:
            prediction = 'Stone'
            return [prediction}]
        elif result[0] == 3:
            prediction = 'Tumor'
            return [prediction}]
        else:
            prediction = 'Unknown'
            return [prediction}]
