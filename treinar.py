import os
import json
import cv2
import numpy as np

dataset_root = "data"
model_path = "model.yml"
labels_path = "labels.json"

if not os.path.isdir(dataset_root):
    print("Pasta 'data' não encontrada. Execute o captura.py primeiro.")
    exit(1)

# Mapeia nomes -> IDs
label_to_id = {}
images = []
labels = []
current_id = 0

for nome in sorted(os.listdir(dataset_root)):
    pessoa_dir = os.path.join(dataset_root, nome)
    if not os.path.isdir(pessoa_dir):
        continue

    if nome not in label_to_id:
        label_to_id[nome] = current_id
        current_id += 1

    for file in os.listdir(pessoa_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(pessoa_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # garante tamanho consistente
        img = cv2.resize(img, (200, 200))
        images.append(img)
        labels.append(label_to_id[nome])

if not images:
    print("Nenhuma imagem encontrada. Capture rostos com captura.py.")
    exit(1)

# Treina LBPH (precisa de opencv-contrib-python)
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train(images, np.array(labels))
recognizer.save(model_path)

# salva labels
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=2)

print(f"Modelo salvo em: {model_path}")
print(f"Labels salvos em: {labels_path}")
