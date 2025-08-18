import cv2
import mediapipe as mp
import numpy as np
import json

MODEL_PATH = "modelo.yml"
LABELS_PATH = "labels.json"
LIMITE_CONF = 70

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)  
except FileNotFoundError:
    print(f"Arquivo {LABELS_PATH} não encontrado. Execute primeiro treinar.py")
    exit(1)

labels_inv = {v: k for k, v in labels.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

mp_face_detection = mp.solutions.face_detection

ultima_posicao = None  

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
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            m = 20
            x1 = max(0, x - m)
            y1 = max(0, y - m)
            x2 = min(w, x + bw + m)
            y2 = min(h, y + bh + m)

            if ultima_posicao is not None:
                dx = abs(x1 - ultima_posicao[0])
                dy = abs(y1 - ultima_posicao[1])
                if dx < 30 and dy < 30:
                    x1, y1, x2, y2 = ultima_posicao
                else:
                    ultima_posicao = (x1, y1, x2, y2)
            else:
                ultima_posicao = (x1, y1, x2, y2)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(gray, (200, 200))
                id_, conf = recognizer.predict(face_resized)

                if conf > LIMITE_CONF:
                    nome = "Desconhecido"
                    cor = (0, 0, 255) 
                else:
                    nome = labels_inv.get(id_, "Desconhecido")
                    cor = (0, 255, 0)  

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, f"{nome} ({int(conf)})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

        cv2.imshow("Reconhecimento Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
