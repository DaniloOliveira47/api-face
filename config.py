import os

FACES_FOLDER = 'faces'
os.makedirs(FACES_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_FILE = 'face_model.pkl'
PCA_THRESHOLD = 800
DIRECT_THRESHOLD = 1500
MIN_FACE_SIZE = 100
