from flask import Blueprint, send_from_directory
from config import FACES_FOLDER
import os

util_bp = Blueprint('util', __name__)

@util_bp.route('/foto/<filename>', methods=['GET'])
def get_foto(filename):
    path = os.path.join(FACES_FOLDER, filename)
    if not os.path.exists(path):
        return {"erro": "Foto não encontrada"}, 404
    return send_from_directory(FACES_FOLDER, filename)
