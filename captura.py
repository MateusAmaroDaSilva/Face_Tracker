import cv2
import mediapipe as mp
import numpy as np
import io
from supabase import create_client, Client
import time
import os

# Configurações Supabase
SUPABASE_URL = "https://bngwnknyxmhkeesoeizb.supabase.co"
SUPABASE_KEY = "SUA_CHAVE_AQUI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


pessoa = input("Digite seu Nome: ").strip()
if not pessoa:
    print("Nome inválido.")
    exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit(1)

mp_face_detection = mp.solutions.face_detection

print("Abra a câmera, enquadre o rosto. Pressione 'S' para salvar, 'Q' para sair.")

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Captura de Rosto", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s') and results.detections:
            face = frame[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (160, 160)) 
            face_normalized = face_resized.astype(np.float32) / 255.0
            embedding = face_normalized.flatten()  

            filename = f"embedding_{int(time.time()*1000)}.npy"
            path_in_bucket = f"{pessoa}/{filename}"

            # Converter para bytes e enviar
            file_bytes = io.BytesIO()
            np.save(file_bytes, embedding)
            file_bytes.seek(0)
            supabase.storage.from_("faces").upload(path_in_bucket, file_bytes.read())

            print(f"Embedding salvo no Supabase: {path_in_bucket}")

cap.release()
cv2.destroyAllWindows()
print("Captura finalizada.")
