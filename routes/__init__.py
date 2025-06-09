from flask import Blueprint

from .setor_routes import setor_bp
from .pessoa_routes import pessoa_bp
from .reconhecimento_routes import reconhecimento_bp
from .util_routes import util_bp

def register_routes(app):
    app.register_blueprint(setor_bp)
    app.register_blueprint(pessoa_bp)
    app.register_blueprint(reconhecimento_bp)
    app.register_blueprint(util_bp)
