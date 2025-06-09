import uuid
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Setor(db.Model):
    __tablename__ = 'setores'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = db.Column(db.String(100), nullable=False, unique=True)
    descricao = db.Column(db.String(255))
    data_criacao = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "descricao": self.descricao,
            "data_criacao": self.data_criacao.isoformat()
        }


class Pessoa(db.Model):
    __tablename__ = 'pessoas'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = db.Column(db.String(100), nullable=False)
    setor_id = db.Column(db.String(36), db.ForeignKey('setores.id'), nullable=False)
    setor = db.relationship('Setor', backref='pessoas')
    foto = db.Column(db.String(100), nullable=False)
    face_data = db.Column(db.LargeBinary(length=16777215))
    data_cadastro = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "setor": self.setor.to_dict() if self.setor else None,
            "foto": self.foto,
            "data_cadastro": self.data_cadastro.isoformat()
        }


class Pasta(db.Model):
    __tablename__ = 'pastas'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = db.Column(db.String(100), nullable=False)
    setor_id = db.Column(db.String(36), db.ForeignKey('setores.id'), nullable=True)  # Apenas para pastas raiz
    setor = db.relationship('Setor', backref='pastas')
    pasta_pai_id = db.Column(db.String(36), db.ForeignKey('pastas.id'), nullable=True)
    pasta_pai = db.relationship('Pasta', remote_side=[id], backref='subpastas')
    data_criacao = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "setor_id": self.setor_id,
            "pasta_pai_id": self.pasta_pai_id,
            "data_criacao": self.data_criacao.isoformat()
        }


class Arquivo(db.Model):
    __tablename__ = 'arquivos'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    nome = db.Column(db.String(100), nullable=False)
    caminho = db.Column(db.String(255), nullable=False)  # path físico ou lógico
    pasta_id = db.Column(db.String(36), db.ForeignKey('pastas.id'), nullable=False)
    pasta = db.relationship('Pasta', backref='arquivos')
    data_upload = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "caminho": self.caminho,
            "pasta_id": self.pasta_id,
            "data_upload": self.data_upload.isoformat()
        }
