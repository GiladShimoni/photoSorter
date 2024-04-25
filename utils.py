from .imports import *



def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image



