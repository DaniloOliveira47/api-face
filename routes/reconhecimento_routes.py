from flask import Blueprint, request, jsonify
from face_utils import allowed_file, detect_single_face, preprocess_face, compare_faces
from models import Pessoa
from pca_utils import load_pca_model
import numpy as np
import cv2

reconhecimento_bp = Blueprint('reconhecimento', __name__)
pca_model = load_pca_model()

@reconhecimento_bp.route('/reconhecer', methods=['POST'])
def reconhecer():
    imagem = request.files.get('imagem')
    if not imagem or not allowed_file(imagem.filename):
        return jsonify({'erro': 'Imagem inválida'}), 400

    image_bytes = np.frombuffer(imagem.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    face_image = detect_single_face(image)
    input_face_vector = preprocess_face(face_image)

    if input_face_vector is None:
        return jsonify({'erro': 'Rosto não detectado'}), 400

    pessoas = Pessoa.query.all()
    if not pessoas:
        return jsonify({'erro': 'Nenhuma pessoa cadastrada'}), 404

    reconhecida = None
    menor_distancia = float('inf')
    for p in pessoas:
        face_vector = pickle.loads(p.face_data)
        distancia, metodo, limite = compare_faces(input_face_vector, face_vector, pca_model)
        if distancia < limite and distancia < menor_distancia:
            menor_distancia = distancia
            reconhecida = p

    if reconhecida:
        return jsonify({'sucesso': True, 'pessoa': reconhecida.to_dict(), 'distancia': menor_distancia})
    else:
        return jsonify({'erro': 'Pessoa não reconhecida'})
