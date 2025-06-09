import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from models import Pessoa
from config import MODEL_FILE
from flask import current_app as app

pca = None

def update_pca_model(db):
    try:
        pessoas = Pessoa.query.all()
        if len(pessoas) >= 5:
            X = np.array([pickle.loads(p.face_data) for p in pessoas])
            n_components = min(50, len(pessoas)-1)
            global pca
            pca = PCA(n_components=n_components)
            pca.fit(X)
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(pca, f)
            app.logger.info(f"PCA treinado com {n_components} componentes.")
    except Exception as e:
        app.logger.error(f"Erro PCA: {str(e)}")

def load_pca_model():
    global pca
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                pca = pickle.load(f)
        except Exception as e:
            app.logger.error(f"Erro ao carregar PCA: {str(e)}")
    return pca
