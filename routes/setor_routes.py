from flask import Blueprint, request, jsonify
from models import db, Setor
import uuid

setor_bp = Blueprint('setor', __name__)

@setor_bp.route('/setores', methods=['GET'])
def listar_setores():
    setores = Setor.query.all()
    return jsonify({"setores": [s.to_dict() for s in setores], "total": len(setores)})

@setor_bp.route('/setor', methods=['POST'])
def criar_setor():
    data = request.get_json()
    if not data or 'nome' not in data:
        return jsonify({"erro": "Nome do setor é obrigatório"}), 400
    if Setor.query.filter_by(nome=data['nome']).first():
        return jsonify({"erro": "Setor já existe"}), 400

    setor = Setor(id=str(uuid.uuid4()), nome=data['nome'], descricao=data.get('descricao', ''))
    db.session.add(setor)
    db.session.commit()
    return jsonify({"sucesso": True, "setor": setor.to_dict()})

@setor_bp.route('/setor/<string:setor_id>', methods=['DELETE'])
def deletar_setor(setor_id):
    setor = Setor.query.get(setor_id)
    if not setor:
        return jsonify({"erro": "Setor não encontrado"}), 404
    
    try:
        # Primeiro deleta todas as pessoas relacionadas ao setor
        for pessoa in setor.pessoas:
            db.session.delete(pessoa)
        
        # Depois deleta o setor
        db.session.delete(setor)
        db.session.commit()
        return jsonify({
            "sucesso": True,
            "mensagem": f"Setor '{setor.nome}' e todas as pessoas vinculadas foram deletados com sucesso"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "erro": f"Não foi possível deletar o setor: {str(e)}"
        }), 500
    
    