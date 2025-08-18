# Face Tracker - Sistema de Reconhecimento Facial

Este projeto é um sistema de reconhecimento facial em **tempo real**, desenvolvido em Python, utilizando OpenCV e MediaPipe, sem necessidade de `dlib`. Ideal para estudos e portfólio.

---

## 📁 Estrutura do Projeto
```bash
📂 reconhecimento_facial/
│── 📄 capturar.py        # Captura imagens do usuário e envia para o bucket no Supabase
│── 📄 treinar.py         # Baixa as imagens do Supabase, treina o modelo e gera modelo.yml + labels.txt
│── 📄 reconhecer.py      # Carrega modelo.yml e labels.txt e faz o reconhecimento em tempo real
│── 📄 modelo.yml         # Arquivo do modelo treinado (gerado pelo treinar.py)
│── 📄 labels.txt         # Mapeamento ID -> Nome completo (gerado pelo treinar.py)
│── 📄 requirements.txt   # Dependências (cv2, mediapipe, supabase, numpy etc.)
```

---

## 💻 Instalação

1. Instale o [Python 3.12](https://www.python.org/downloads/) (marque "Add Python to PATH").  
2. Instale o [VS Code](https://code.visualstudio.com/) e a extensão **Python**.  
3. Abra o terminal na pasta do projeto e crie um ambiente virtual:

```bash
python -m venv venv
.\venv\Scripts\activate
```
4. **Instale as dependências**:
```bash
pip install -r requirements.txt
```
Conteúdo do `requirements.txt`:
```bash
opencv-contrib-python
mediapipe
numpy
```
---

## Como Rodar:
1. **Capturar Rostos**:

*Execute*:
```bash
python captura.py
```
- **Digite o nome da pessoa**.
- **A webcam abrirá e detectará o rosto**.
- **Comandos durante a captura**:
    - S → salvar rosto detectado (em `data/<nome>/`)
    - Q → sair do programa

Capturar pelo menos **30 fotos por pessoa**, variando ângulos, expressões e iluminação.

---

2. **Treinar o Modelo**:

*Execute*:
```bash
python treinar.py
```
- O script lê todas as fotos em `data/.`.
- onverte para tons de cinza e tamanho fixo (200x200 px).
- Treina o modelo **LBPH** e salva:
    - `model.yml` → modelo treinado
    - `labels.json` → mapeamento de nomes e IDs

Só precisa treinar novamente se adicionar novas pessoas.

---

3. **Reconhecer em Tempo Real**:

*Execute*:
```bash
python reconhecer.py
```
- A webcam abre novamente.
- Detecta rostos usando MediaPipe e reconhece usando **LBPH**.
- **Resultados**:
    - Retângulo **verde** → pessoa reconhecida
    - Retângulo **vermelho** → desconhecido
- Pressione **Q** para sair.

Ajuste a variável `THRESHOLD` dentro do script para aumentar ou diminuir a sensibilidade do reconhecimento.

---

🛠️ Conclusão

O **Face Tracker** é um sistema de reconhecimento facial funcional e educativo, **ainda em desenvolvimento**. Futuras melhorias podem incluir:

- Reconhecimento de múltiplas pessoas simultaneamente.
- Armazenamento de logs e histórico de reconhecimento.
- Melhorias na interface gráfica.
- Ajustes na sensibilidade e precisão do modelo LBPH.

Este projeto é uma base sólida para quem quer aprender sobre visão computacional e reconhecimento facial em **Python**.
