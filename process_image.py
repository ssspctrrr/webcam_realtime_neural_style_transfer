import tensorflow_hub as hub
import tensorflow as tf
import cv2

class ProcessImage:
    # Load the pre-trained model from TensorFlow Hub
    __hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    __hub_module = hub.load(__hub_handle)

    # Style transfer function with softer effect
    def apply_style_transfer(self, content_image, style_image, alpha=0.5):
        content_tensor = tf.convert_to_tensor(content_image, dtype=tf.float32)
        style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
        content_tensor = tf.expand_dims(content_tensor, axis=0)
        style_tensor = tf.expand_dims(style_tensor, axis=0)
        outputs = self.__hub_module(content_tensor, style_tensor)
        stylized_image = outputs[0].numpy()[0]
        # Blend the original content image with the stylized image
        blended_image = alpha * stylized_image + (1 - alpha) * content_image
        return blended_image

    # Preprocess image
    def preprocess_image(self, image, target_size=(256, 256)):
        resized_image = cv2.resize(image, target_size)
        return resized_image / 255.0  # Normalize to [0, 1]