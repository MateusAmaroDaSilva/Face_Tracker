import cv2
import mediapipe as mp
import numpy as np
import io
import json
from supabase import create_client

SUPABASE_URL = "https://bngwnknyxmhkeesoeizb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJuZ3dua255eG1oa2Vlc29laXpiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU1MzUwMDcsImV4cCI6MjA3MTExMTAwN30.MVVHAuicG_pkv0OR1h3HEwI-gx7d5hYoqX-xrK17B_U" 
BUCKET_NAME = "faces"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LIMITE_CONF = 0.8 

LABELS_PATH = "labels.json"
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
except FileNotFoundError:
    print(f"Arquivo {LABELS_PATH} não encontrado. Execute primeiro treinar.py")
    exit(1)

labels_inv = {v: k for k, v in labels.items()}

def similaridade(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    if v1.shape != v2.shape:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Buscando embeddings no Supabase...")
embeddings = {}
for nome_usuario in labels.keys():
    arquivos = supabase.storage.from_(BUCKET_NAME).list(path=nome_usuario)
    if not arquivos:
        continue

    embeddings_usuario = []
    for arquivo in arquivos:
        if arquivo['name'].endswith(".npy"):
            caminho_bucket = f"{nome_usuario}/{arquivo['name']}"
            try:
                arquivo_bytes = supabase.storage.from_(BUCKET_NAME).download(caminho_bucket)
                emb = np.load(io.BytesIO(arquivo_bytes))
                embeddings_usuario.append(emb)
            except Exception as e:
                print(f" Erro ao baixar {caminho_bucket}: {e}")

    if embeddings_usuario:
        embeddings[nome_usuario] = embeddings_usuario
        print(f"Deu Cert!!! Embedding carregado: {nome_usuario}/{arquivos[-1]['name']}")

if not embeddings:
    print("Nenhum embedding encontrado.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit(1)

mp_face_detection = mp.solutions.face_detection

print("Pressione 'Q' para sair.")
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w) - 20)
            y1 = max(0, int(bbox.ymin * h) - 20)
            x2 = min(w, int((bbox.xmin + bbox.width) * w) + 20)
            y2 = min(h, int((bbox.ymin + bbox.height) * h) + 20)

            face = frame[y1:y2, x1:x2]
            nome = "Desconhecido"
            cor = (0, 0, 255)  

            if face.size > 0:
                face_resized = cv2.resize(face, (160, 160))
                face_normalized = face_resized.astype(np.float32) / 255.0
                face_embedding = face_normalized.flatten()

                max_sim = -1
                for usuario, embs in embeddings.items():
                    for emb in embs:
                        sim = similaridade(face_embedding, emb)
                        if sim > max_sim:
                            max_sim = sim
                            if sim > LIMITE_CONF:
                                nome = usuario
                                cor = (0, 255, 0)  

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, nome, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
