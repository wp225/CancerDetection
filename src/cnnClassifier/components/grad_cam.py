from typing import List
import keras.models
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import config
from matplotlib.pyplot import get_cmap as cm


class GradCam:

    def __init__(self, model, last_conv_layer: str, classifier_layers: List[str], processed_image, original_image):
        self.model: Model = model
        self.last_conv_layer = last_conv_layer
        self.classifier_layers = classifier_layers
        self.original_image = keras.preprocessing.image.img_to_array(original_image)
        self.processed_image = processed_image

    def cam(self):
        last_conv_layer = self.model.get_layer(self.last_conv_layer)
        last_conv_model = Model(inputs=self.model.input, outputs=last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        for layer_name in self.classifier_layers:
            x = self.model.get_layer(layer_name)(x)

        classifier_model = Model(inputs=classifier_input, outputs=x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_model(self.processed_image)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = np.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grad = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grad = tf.reduce_mean(grad, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grad = pooled_grad.numpy()

        for i in range(pooled_grad.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grad[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        heatmap = np.uint8(255 * heatmap)

        jet = cm('jet')
        jet_color = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_color[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((self.original_image.shape[1], self.original_image.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + self.original_image

        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        return superimposed_img


if __name__ == '__main__':
    import os
    from tensorflow.keras.preprocessing import image

    last_conv_layer_name = 'block5_conv3'
    classifier_layer_names = ['global_average_pooling2d', 'dense', 'dense_1']
    model = keras.models.load_model('artifacts/training/model.h5')
    imagename = 'artifacts/data_ingestion/Chest-CT-Scan-data/adenocarcinoma/000000 (6).png'
    original_image = image.load_img(imagename, target_size=(224, 224))
    test_image = image.img_to_array(original_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = tf.keras.applications.vgg16.preprocess_input(test_image)  # Ensure preprocessing matches the model

    obj = GradCam(model, last_conv_layer_name, classifier_layer_names, test_image, original_image)
    heatmap = obj.cam()