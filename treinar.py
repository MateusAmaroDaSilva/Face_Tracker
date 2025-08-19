import numpy as np
import json
import io
from supabase import create_client

# Configurações Supabase
SUPABASE_URL = "https://bngwnknyxmhkeesoeizb.supabase.co"
SUPABASE_KEY = "SEU_SUPABASE_KEY_AQUI"
BUCKET_NAME = "faces"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Baixando embeddings do bucket Supabase...")

usuarios = supabase.storage.from_(BUCKET_NAME).list(path="")
if not usuarios:
    print(" Nenhum usuário encontrado no bucket.")
    exit()

embeddings = []
labels = []
label_ids = {}
current_id = 0

for user in usuarios:
    nome_usuario = user['name']

    try:
        arquivos = supabase.storage.from_(BUCKET_NAME).list(path=nome_usuario)
    except Exception as e:
        print(f" Erro ao listar {nome_usuario}: {e}")
        continue

    arquivos_npy = [a['name'] for a in arquivos if a['name'].endswith('.npy')]
    if not arquivos_npy:
        print(f" Nenhum embedding encontrado para {nome_usuario}, pulando...")
        continue

    for arquivo_npy in arquivos_npy:
        caminho_bucket = f"{nome_usuario}/{arquivo_npy}"
        try:
            arquivo_bytes = supabase.storage.from_(BUCKET_NAME).download(caminho_bucket)
            vetor = np.load(io.BytesIO(arquivo_bytes))
            embeddings.append(vetor)
            if nome_usuario not in label_ids:
                label_ids[nome_usuario] = current_id
                current_id += 1
            labels.append(label_ids[nome_usuario])
        except Exception as e:
            print(f" Erro ao baixar {caminho_bucket}: {e}")
            continue

if not embeddings:
    print(" Nenhum embedding válido encontrado. Abortando treino.")
    exit()

print(f" {len(embeddings)} embeddings carregados para treino.")

# Salvar labels.json atualizado
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(label_ids, f, ensure_ascii=False, indent=4)

print("Labels atualizados em 'labels.json':")
print(label_ids)
