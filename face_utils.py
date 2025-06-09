import cv2
import numpy as np
from config import MIN_FACE_SIZE, PCA_THRESHOLD, DIRECT_THRESHOLD

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def validate_image(file):
    if not file or file.filename == '':
        return False, "Nenhum arquivo selecionado"
    if not allowed_file(file.filename):
        return False, "Tipo de arquivo não permitido"
    return True, ""

def detect_single_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
    if len(faces) == 0:
        return None
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
    return gray[y:y+h, x:x+w]

def preprocess_face(face_image):
    if face_image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_image = clahe.apply(face_image)
    face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
    face_image = cv2.resize(face_image, (100, 100))
    face_image = face_image.astype(np.float32) / 255.0
    return face_image.flatten()

def compare_faces(face1, face2, pca_model=None):
    if pca_model:
        face1_proj = pca_model.transform([face1])[0]
        face2_proj = pca_model.transform([face2])[0]
        distance = np.linalg.norm(face1_proj - face2_proj)
        return distance, "PCA", PCA_THRESHOLD
    else:
        distance = np.linalg.norm(face1 - face2)
        return distance, "Direto", DIRECT_THRESHOLD
