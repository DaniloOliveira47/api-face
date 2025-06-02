from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
from sklearn.decomposition import PCA
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Permite todas as origens (apenas para desenvolvimento)

# Configurações
FACES_FOLDER = 'faces'
os.makedirs(FACES_FOLDER, exist_ok=True)

# Extensões permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Carregar classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Banco de dados e modelo PCA
database = []
pca = None
model_file = 'face_model.pkl'

# Thresholds ajustados
PCA_THRESHOLD = 800  # Ajuste conforme necessidade
DIRECT_THRESHOLD = 1500  # Ajuste conforme necessidade
MIN_FACE_SIZE = 100  # Tamanho mínimo do rosto em pixels

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Valida a imagem antes do processamento"""
    if not file or file.filename == '':
        return False, "Nenhum arquivo selecionado"
    
    if not allowed_file(file.filename):
        return False, "Tipo de arquivo não permitido"
    
    return True, ""

def detect_single_face(image):
    """Detecta um único rosto na imagem"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Parâmetros ajustados para melhor detecção
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Retorna o maior rosto detectado
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
    return gray[y:y+h, x:x+w]

def preprocess_face(face_image):
    """Pré-processamento consistente para todas as imagens"""
    if face_image is None:
        return None
    
    # Equalização de histograma adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_image = clahe.apply(face_image)
    
    # Suavização para reduzir ruído
    face_image = cv2.GaussianBlur(face_image, (3, 3), 0)
    
    # Redimensionar e normalizar
    face_image = cv2.resize(face_image, (100, 100))
    face_image = face_image.astype(np.float32) / 255.0  # Normalizar para [0, 1]
    
    return face_image.flatten()

def save_face_data(pessoa, face_data):
    """Salva os dados da pessoa no banco de dados"""
    pessoa["face_data"] = face_data.tolist()
    database.append(pessoa)
    
    # Atualizar PCA se tiver dados suficientes
    update_pca_model()

def update_pca_model():
    """Atualiza o modelo PCA quando há dados suficientes"""
    if len(database) >= 5:
        try:
            X = np.array([p['face_data'] for p in database])
            n_components = min(50, len(database)-1)
            global pca
            pca = PCA(n_components=n_components)
            pca.fit(X)
            
            with open(model_file, 'wb') as f:
                pickle.dump(pca, f)
                
            app.logger.info(f"Modelo PCA atualizado com {n_components} componentes")
        except Exception as e:
            app.logger.error(f"Erro ao atualizar PCA: {str(e)}")

def load_pca_model():
    """Carrega o modelo PCA se existir"""
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            app.logger.error(f"Erro ao carregar PCA: {str(e)}")
    return None

def compare_faces(face1, face2, pca_model=None):
    """Compara duas faces usando PCA ou distância direta"""
    if pca_model:
        # Projeção PCA
        face1_proj = pca_model.transform([face1])[0]
        face2_proj = pca_model.transform([face2])[0]
        distance = np.linalg.norm(face1_proj - face2_proj)
        method = "PCA"
        threshold = PCA_THRESHOLD
    else:
        # Comparação direta
        distance = np.linalg.norm(face1 - face2)
        method = "Direto"
        threshold = DIRECT_THRESHOLD
    
    return distance, method, threshold

# Rota para servir imagens
@app.route('/faces/<filename>')
def uploaded_file(filename):
    return send_from_directory(FACES_FOLDER, secure_filename(filename))

# Rota para cadastrar nova pessoa
@app.route('/cadastrar', methods=['POST'])
def cadastrar():
    if 'foto' not in request.files:
        return jsonify({"erro": "Nenhuma foto enviada"}), 400

    file = request.files['foto']
    valid, message = validate_image(file)
    if not valid:
        return jsonify({"erro": message}), 400

    try:
        # Ler e validar imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"erro": "Não foi possível ler a imagem"}), 400

        # Detectar rosto
        face_roi = detect_single_face(img)
        if face_roi is None:
            return jsonify({"erro": "Nenhum rosto detectado ou rosto muito pequeno"}), 400

        # Pré-processamento
        face_data = preprocess_face(face_roi)
        if face_data is None:
            return jsonify({"erro": "Erro no pré-processamento da face"}), 500

        # Criar registro
        pessoa_id = str(uuid.uuid4())
        filename = f"{pessoa_id}.jpg"
        cv2.imwrite(os.path.join(FACES_FOLDER, filename), img)
        
        pessoa = {
            "id": pessoa_id,
            "nome": request.form.get('nome', '').strip(),
            "setor": request.form.get('setor', '').strip(),
            "foto": filename
        }
        
        save_face_data(pessoa, face_data)
        
        return jsonify({
            "sucesso": True,
            "pessoa": {
                "id": pessoa_id,
                "nome": pessoa['nome'],
                "setor": pessoa['setor'],
                "foto": filename
            }
        })

    except Exception as e:
        app.logger.error(f"Erro no cadastro: {str(e)}", exc_info=True)
        return jsonify({"erro": "Erro interno no processamento"}), 500

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    if 'foto' not in request.files:
        return jsonify({"erro": "Nenhuma foto enviada"}), 400

    file = request.files['foto']
    valid, message = validate_image(file)
    if not valid:
        return jsonify({"erro": message}), 400

    try:
        # Ler e validar imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"erro": "Não foi possível ler a imagem"}), 400

        # Detectar rosto
        face_roi = detect_single_face(img)
        if face_roi is None:
            return jsonify({"erro": "Nenhum rosto detectado ou rosto muito pequeno"}), 400

        # Pré-processamento
        face_test = preprocess_face(face_roi)
        if face_test is None:
            return jsonify({"erro": "Erro no pré-processamento da face"}), 500

        # Verificar se há cadastros
        if not database:
            return jsonify({"erro": "Nenhuma pessoa cadastrada"}), 404

        # Carregar PCA se existir
        pca_model = load_pca_model() if len(database) >= 5 else None

        # Comparar com todas as faces cadastradas
        resultados = []
        for pessoa in database:
            try:
                registered_face = np.array(pessoa['face_data'], dtype=np.float32)
                
                # Verificar consistência dimensional
                if len(face_test) != len(registered_face):
                    app.logger.warning(f"Dimensões incompatíveis: {len(face_test)} vs {len(registered_face)}")
                    continue
                
                distance, method, threshold = compare_faces(face_test, registered_face, pca_model)
                
                # Converter numpy.bool_ para bool nativo do Python
                is_match = bool(distance < threshold)
                
                resultados.append({
                    "pessoa": {
                        "id": pessoa["id"],
                        "nome": pessoa["nome"],
                        "setor": pessoa["setor"],
                        "foto": pessoa["foto"]
                    },
                    "distancia": float(distance),
                    "metodo": method,
                    "threshold": float(threshold),
                    "match": is_match
                })
            except Exception as e:
                app.logger.error(f"Erro ao comparar com {pessoa.get('nome', 'desconhecido')}: {str(e)}")
                continue

        if not resultados:
            return jsonify({"erro": "Nenhuma comparação realizada"}), 500

        # Ordenar por menor distância
        resultados.sort(key=lambda x: x['distancia'])
        melhor_match = resultados[0]
        
        # Preparar resposta
        response_data = {
            "sucesso": True if melhor_match['match'] else False,
            "melhor_match": melhor_match,
            "debug": {
                "pca_used": pca_model is not None,
                "total_cadastros": len(database)
            }
        }
        
        # Adicionar todas as comparações apenas se estivermos em modo debug
        if app.debug:
            response_data["todas_comparacoes"] = resultados

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Erro no reconhecimento: {str(e)}", exc_info=True)
        return jsonify({"erro": "Erro interno no processamento"}), 500
@app.route('/pessoas', methods=['GET'])
def listar_pessoas():
    return jsonify({
        "pessoas": [{
            "id": p["id"],
            "nome": p["nome"],
            "setor": p["setor"],
            "foto": p["foto"]
        } for p in database],
        "total": len(database)
    })

@app.route('/pessoa/<pessoa_id>', methods=['GET'])
def buscar_pessoa(pessoa_id):
    pessoa = next((p for p in database if p['id'] == pessoa_id), None)
    if pessoa:
        return jsonify({
            "id": pessoa["id"],
            "nome": pessoa["nome"],
            "setor": pessoa["setor"],
            "foto": pessoa["foto"]
        })
    return jsonify({"erro": "Pessoa não encontrada"}), 404

@app.route('/limpar', methods=['POST'])
def limpar_banco():
    global database, pca
    database = []
    pca = None
    try:
        if os.path.exists(model_file):
            os.remove(model_file)
        for file in os.listdir(FACES_FOLDER):
            os.remove(os.path.join(FACES_FOLDER, file))
        return jsonify({"sucesso": True, "mensagem": "Banco de dados limpo"})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)