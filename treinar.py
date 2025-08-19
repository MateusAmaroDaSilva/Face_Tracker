import cv2
import numpy as np
from supabase import create_client
import json

# Configurações Supabase
SUPABASE_URL = "https://bngwnknyxmhkeesoeizb.supabase.co"
SUPABASE_KEY = "SUA_CHAVE_AQUI"
BUCKET_NAME = "faces"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🔄 Baixando imagens do bucket Supabase...")

# Listar usuários (pastas)
usuarios = supabase.storage.from_(BUCKET_NAME).list(path="")
if not usuarios:
    print(" Nenhum usuário encontrado no bucket.")
    exit()

imagens = []
labels = []
label_ids = {}
current_id = 0

for user in usuarios:
    nome_usuario = user['name']
    arquivos = supabase.storage.from_(BUCKET_NAME).list(path=nome_usuario)

    for arquivo in arquivos:
        caminho_bucket = f"{nome_usuario}/{arquivo['name']}"
        try:
            arquivo_bytes = supabase.storage.from_(BUCKET_NAME).download(caminho_bucket)
        except Exception as e:
            print(f" Erro ao baixar {caminho_bucket}: {e}")
            continue

        if arquivo_bytes is None or len(arquivo_bytes) == 0:
            continue

        nparr = np.frombuffer(arquivo_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imagens.append(img)
            if nome_usuario not in label_ids:
                label_ids[nome_usuario] = current_id
                current_id += 1
            labels.append(label_ids[nome_usuario])

if not imagens:
    print(" Nenhuma imagem válida encontrada. Abortando treino.")
    exit()

print(f"✅ {len(imagens)} imagens carregadas para treino.")

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print(" Módulo 'cv2.face' não encontrado. Instale 'opencv-contrib-python'.")
    exit()

# Treinar modelo
recognizer.train(imagens, np.array(labels))

# Salvar modelo
recognizer.write("modelo.yml")
print("O treinamento foi um SUCESSO! Modelo salvo em 'modelo.yml'.")

# Salvar labels para uso futuro
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(label_ids, f, ensure_ascii=False, indent=4)

print("Labels salvos em 'labels.json':")
print(label_ids)
