from flask import Blueprint, request, jsonify
from models import db, Pessoa, Setor
from face_utils import allowed_file, detect_single_face, preprocess_face
from pca_utils import update_pca_model
import uuid
import os
import cv2
import pickle
import numpy as np
from config import FACES_FOLDER

pessoa_bp = Blueprint('pessoa', __name__)

@pessoa_bp.route('/pessoa', methods=['POST'])
def cadastrar_pessoa():
    nome = request.form.get('nome')
    setor_id = request.form.get('setor_id')
    foto = request.files.get('foto')

    if not nome or not setor_id or not foto:
        return jsonify({'erro': 'Campos obrigatórios não informados'}), 400

    if not allowed_file(foto.filename):
        return jsonify({'erro': 'Formato de imagem inválido'}), 400

    setor = Setor.query.get(setor_id)
    if not setor:
        return jsonify({'erro': 'Setor não encontrado'}), 404

    image_bytes = np.frombuffer(foto.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    face_image = detect_single_face(image)
    face_vector = preprocess_face(face_image)

    if face_vector is None:
        return jsonify({'erro': 'Rosto não detectado'}), 400

    pessoa_id = str(uuid.uuid4())
    filename = f"{pessoa_id}.jpg"
    path = os.path.join(FACES_FOLDER, filename)
    cv2.imwrite(path, image)

    pessoa = Pessoa(
        id=pessoa_id,
        nome=nome,
        setor_id=setor_id,
        foto=filename,
        face_data=pickle.dumps(face_vector)
    )
    db.session.add(pessoa)
    db.session.commit()

    update_pca_model(db)

    return jsonify({'sucesso': True, 'pessoa': pessoa.to_dict()})

@pessoa_bp.route('/pessoas', methods=['GET'])
def listar_pessoas():
    pessoas = Pessoa.query.all()
    return jsonify({'pessoas': [p.to_dict() for p in pessoas], 'total': len(pessoas)})

@pessoa_bp.route('/pessoa/<string:pessoa_id>', methods=['PUT'])
def atualizar_pessoa(pessoa_id):
    pessoa = Pessoa.query.get(pessoa_id)
    if not pessoa:
        return jsonify({'erro': 'Pessoa não encontrada'}), 404

    nome = request.form.get('nome')
    setor_id = request.form.get('setor_id')
    foto = request.files.get('foto')

    if nome:
        pessoa.nome = nome
    
    if setor_id:
        setor = Setor.query.get(setor_id)
        if not setor:
            return jsonify({'erro': 'Setor não encontrado'}), 404
        pessoa.setor_id = setor_id

    if foto:
        if not allowed_file(foto.filename):
            return jsonify({'erro': 'Formato de imagem inválido'}), 400

        # Remove a foto antiga
        old_path = os.path.join(FACES_FOLDER, pessoa.foto)
        if os.path.exists(old_path):
            os.remove(old_path)

        # Processa a nova foto
        image_bytes = np.frombuffer(foto.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        face_image = detect_single_face(image)
        face_vector = preprocess_face(face_image)

        if face_vector is None:
            return jsonify({'erro': 'Rosto não detectado na nova imagem'}), 400

        # Salva a nova foto
        filename = f"{pessoa_id}.jpg"
        path = os.path.join(FACES_FOLDER, filename)
        cv2.imwrite(path, image)
        
        pessoa.foto = filename
        pessoa.face_data = pickle.dumps(face_vector)

    db.session.commit()
    
    if foto:
        update_pca_model(db)

    return jsonify({'sucesso': True, 'pessoa': pessoa.to_dict()})

@pessoa_bp.route('/pessoa/<string:pessoa_id>', methods=['DELETE'])
def deletar_pessoa(pessoa_id):
    pessoa = Pessoa.query.get(pessoa_id)
    if not pessoa:
        return jsonify({'erro': 'Pessoa não encontrada'}), 404

    # Remove o arquivo de imagem
    path = os.path.join(FACES_FOLDER, pessoa.foto)
    if os.path.exists(path):
        os.remove(path)

    db.session.delete(pessoa)
    db.session.commit()

    update_pca_model(db)

    return jsonify({'sucesso': True, 'mensagem': 'Pessoa removida com sucesso'})