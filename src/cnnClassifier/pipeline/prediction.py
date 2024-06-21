import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from cnnClassifier.components.grad_cam import GradCam
import config
import matplotlib.pyplot as plt

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
    def _predict(self):
        model = load_model(os.path.join('artifacts','training','model.h5'))

        last_conv_layer_name = 'block5_conv3'
        classifier_layer_names = ['global_average_pooling2d', 'dense', 'dense_1']

        imagename = self.filename
        original_image = image.load_img(imagename,target_size=(224,224))
        test_image = image.img_to_array(original_image)
        test_image = np.expand_dims(test_image,axis=0)
        grad_cam = GradCam(model=model,last_conv_layer=last_conv_layer_name,classifier_layers=classifier_layer_names,
                           original_image=original_image,processed_image=test_image)
        grad_cam_image = grad_cam.cam()
        plt.imshow(grad_cam_image)
        plt.axis('off')
        plt.show()
        result = np.argmax(model.predict(test_image),axis=1)
        if result[0] == 1:
            print('Normal')
        else:
            print('Adinocarsinoma')

if __name__ == '__main__':
    obj = PredictionPipeline('artifacts/data_ingestion/Chest-CT-Scan-data/adenocarcinoma/000005 (3).png')
    obj._predict()
