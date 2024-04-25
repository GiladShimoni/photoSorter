from .imports import *



def detect_faces(image_path):
    """
    This function detects faces in a given image.
    returns a np.array of faces such that the first two elements of each object 
    are the corrdinates of the face, and the third element is the width,
    and the fourth element is the height.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, image



