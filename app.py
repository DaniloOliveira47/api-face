from flask import Flask
from flask_cors import CORS
from models import db
from config import FACES_FOLDER
from routes.setor_routes import setor_bp
from routes.pessoa_routes import pessoa_bp
from routes.reconhecimento_routes import reconhecimento_bp
from routes.util_routes import util_bp
from routes.pastas_routes import bp
import os

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/face_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Registrar rotas
app.register_blueprint(setor_bp)
app.register_blueprint(pessoa_bp)
app.register_blueprint(reconhecimento_bp)
app.register_blueprint(util_bp)
app.register_blueprint(bp)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
