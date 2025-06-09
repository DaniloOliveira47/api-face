from flask import Blueprint, request, jsonify
from models import db, Pasta, Arquivo
import os
import uuid

bp = Blueprint('gerenciador', __name__)

# ✅ Criar nova pasta (raiz ou subpasta)
@bp.route('/pastas', methods=['POST'])
def criar_pasta():
    data = request.json
    nome = data.get('nome')
    setor_id = data.get('setor_id')
    pasta_pai_id = data.get('pasta_pai_id')

    if not nome:
        return jsonify({'erro': 'Nome é obrigatório'}), 400

    nova_pasta = Pasta(
        id=str(uuid.uuid4()),
        nome=nome,
        setor_id=setor_id,
        pasta_pai_id=pasta_pai_id
    )
    db.session.add(nova_pasta)
    db.session.commit()
    return jsonify(nova_pasta.to_dict()), 201

# ✅ Listar todas as pastas
@bp.route('/pastas', methods=['GET'])
def listar_pastas():
    pastas = Pasta.query.all()
    return jsonify([p.to_dict() for p in pastas])

# ✅ Obter pasta por ID
@bp.route('/pastas/<id>', methods=['GET'])
def obter_pasta(id):
    pasta = Pasta.query.get(id)
    if not pasta:
        return jsonify({'erro': 'Pasta não encontrada'}), 404
    return jsonify(pasta.to_dict())

# ✅ Deletar pasta (recursão simples: deletar arquivos e subpastas também)
@bp.route('/pastas/<id>', methods=['DELETE'])
def deletar_pasta(id):
    pasta = Pasta.query.get(id)
    if not pasta:
        return jsonify({'erro': 'Pasta não encontrada'}), 404

    # Deleta arquivos da pasta
    for arquivo in pasta.arquivos:
        db.session.delete(arquivo)

    # Deleta subpastas (recursivamente simples)
    def deletar_subpastas(p):
        for sub in p.subpastas:
            deletar_subpastas(sub)
            for arq in sub.arquivos:
                db.session.delete(arq)
            db.session.delete(sub)

    deletar_subpastas(pasta)
    db.session.delete(pasta)
    db.session.commit()
    return jsonify({'mensagem': 'Pasta deletada com sucesso'})

# ✅ Upload de arquivo (simples, sem arquivo real)
@bp.route('/arquivos', methods=['POST'])
def upload_arquivo():
    data = request.json
    nome = data.get('nome')
    caminho = data.get('caminho')  # Simula path
    pasta_id = data.get('pasta_id')

    if not nome or not caminho or not pasta_id:
        return jsonify({'erro': 'Nome, caminho e pasta_id são obrigatórios'}), 400

    arquivo = Arquivo(
        id=str(uuid.uuid4()),
        nome=nome,
        caminho=caminho,
        pasta_id=pasta_id
    )
    db.session.add(arquivo)
    db.session.commit()
    return jsonify(arquivo.to_dict()), 201

# ✅ Listar arquivos de uma pasta
@bp.route('/arquivos/<pasta_id>', methods=['GET'])
def listar_arquivos_pasta(pasta_id):
    arquivos = Arquivo.query.filter_by(pasta_id=pasta_id).all()
    return jsonify([a.to_dict() for a in arquivos])

# ✅ Deletar arquivo
@bp.route('/arquivos/<id>', methods=['DELETE'])
def deletar_arquivo(id):
    arquivo = Arquivo.query.get(id)
    if not arquivo:
        return jsonify({'erro': 'Arquivo não encontrado'}), 404

    db.session.delete(arquivo)
    db.session.commit()
    return jsonify({'mensagem': 'Arquivo deletado com sucesso'})
