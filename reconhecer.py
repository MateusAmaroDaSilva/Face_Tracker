import cv2
import mediapipe as mp
import json
import os

model_path = "model.yml"
labels_path = "labels.json"

if not (os.path.exists(model_path) and os.path.exists(labels_path)):
    print("Treine primeiro: python treinar.py")
    exit(1)

# carrega modelo e labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
with open(labels_path, "r", encoding="utf-8") as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

mp_face_detection = mp.solutions.face_detection

# Ajuste do limiar de "confiança" do LBPH:
# quanto MENOR o valor retornado, MAIS parecido.
# Comece com 60~70; ajuste conforme seu dataset.
THRESHOLD = 65.0

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
    print("Pressione 'Q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                m = 20
                x1 = max(0, x - m)
                y1 = max(0, y - m)
                x2 = min(w, x + bw + m)
                y2 = min(h, y + bh + m)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))

                label_id, dist = recognizer.predict(gray)
                if dist <= THRESHOLD and label_id in id_to_label:
                    nome = f"{id_to_label[label_id]} ({dist:.1f})"
                    color = (0, 255, 0)
                else:
                    nome = f"Desconhecido ({dist:.1f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, nome, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Reconhecimento (LBPH + MediaPipe)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
