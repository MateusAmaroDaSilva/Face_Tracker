import cv2
import mediapipe as mp
import os
import time

pessoa = input("Digite o NOME da pessoa a cadastrar: ").strip()
if not pessoa:
    print("Nome inválido.")
    exit(1)

dataset_dir = os.path.join("data", pessoa)
os.makedirs(dataset_dir, exist_ok=True)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Não foi possível abrir a webcam.")
    exit(1)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

last_face_gray = None

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
    print("Abra a câmera, enquadre o rosto. Pressione 'S' para SALVAR, 'Q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        last_face_gray = None
        if results.detections:
            det = results.detections[0]
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
            if face.size > 0:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                last_face_gray = cv2.resize(gray, (200, 200))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame, "S: salvar amostra  |  Q: sair",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Captura de Rosto", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and last_face_gray is not None:
            filename = os.path.join(dataset_dir, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, last_face_gray)
            print(f"Salvo: {filename}")

cap.release()
cv2.destroyAllWindows()
print("Captura finalizada.")
