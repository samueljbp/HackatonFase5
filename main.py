import os
import shutil
import smtplib
import ssl
from email.message import EmailMessage
from ultralytics import YOLO
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image, ImageDraw

# ------------------- Configurações -------------------
ADMIN_EMAIL = "admin@example.com"
EMAIL_SENDER = "seu_email@example.com"
EMAIL_PASSWORD = "sua_senha"

# Diretórios
BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(LABELS_TRAIN_DIR, exist_ok=True)
os.makedirs(LABELS_VAL_DIR, exist_ok=True)

# ------------------- Função de envio de e-mail -------------------
def enviar_email(imagem_path, objeto_detectado):
    assunto = "[Alerta] Objeto cortante detectado"
    corpo = f"Foi detectado um(a) {objeto_detectado} na imagem: {imagem_path}"

    msg = EmailMessage()
    msg.set_content(corpo)
    msg["Subject"] = assunto
    msg["From"] = EMAIL_SENDER
    msg["To"] = ADMIN_EMAIL

    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=contexto) as servidor:
        servidor.login(EMAIL_SENDER, EMAIL_PASSWORD)
        servidor.send_message(msg)
    print(f"E-mail enviado para {ADMIN_EMAIL} sobre a detecção de {objeto_detectado}.")

# ------------------- Download e preparação do dataset -------------------
def preparar_dataset():
    class_map = {"Knife": 0, "Scissors": 1, "Sword": 2}
    datasets = []

    for classe in class_map.keys():
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=[classe],
            max_samples=200,
            shuffle=True
        )
        datasets.append(dataset)
    
    # Combina todos os datasets
    dataset_combined = fo.Dataset(name="objetos_cortantes")
    for dataset in datasets:
        dataset_combined.add_samples(dataset)

    for i, sample in enumerate(dataset_combined):
        destino_img = TRAIN_DIR if i % 5 != 0 else VAL_DIR
        destino_lbl = LABELS_TRAIN_DIR if i % 5 != 0 else LABELS_VAL_DIR
        shutil.copy2(sample.filepath, destino_img)

        if sample.ground_truth:
            label_path = os.path.join(destino_lbl, os.path.basename(sample.filepath).replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                for detection in sample.ground_truth.detections:
                    label_idx = class_map.get(detection.label, -1)
                    if label_idx == -1:
                        continue
                    x, y, w, h = detection.bounding_box
                    f.write(f"{label_idx} {x + w / 2} {y + h / 2} {w} {h}\n")
    print(f"Imagens de treino: {len(os.listdir(TRAIN_DIR))}, Imagens de validação: {len(os.listdir(VAL_DIR))}")
    return dataset_combined

# ------------------- Gerar arquivo de configuração do YOLO -------------------
def gerar_data_yaml():
    yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: Knife
  1: Scissors
  2: Sword
"""
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

# ------------------- Treinamento do modelo -------------------
def treinar_modelo():
    gerar_data_yaml()
    modelo = YOLO('yolov8n.pt')
    modelo.train(data=os.path.join(DATASET_DIR, 'data.yaml'), epochs=10, imgsz=640)
    return modelo

# ------------------- Detecção e marcação -------------------
def detectar_objetos(modelo, imagem_path):
    resultados = modelo(imagem_path)
    imagem = Image.open(imagem_path)
    draw = ImageDraw.Draw(imagem)
    objeto_detectado = False

    for resultado in resultados:
        for caixa in resultado.boxes:
            classe = resultado.names[int(caixa.cls)]
            if classe in ["Knife", "Scissors", "Sword"]:
                objeto_detectado = True
                x1, y1, x2, y2 = caixa.xyxy[0]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), classe, fill="red")
    
    imagem_marcada_path = os.path.join(RESULTS_DIR, os.path.basename(imagem_path))
    imagem.save(imagem_marcada_path)

    if objeto_detectado:
        enviar_email(imagem_marcada_path, classe)

# ------------------- Avaliação do modelo -------------------
def avaliar_modelo(modelo, dataset):
    imagens_teste = [sample.filepath for sample in dataset]
    total = len(imagens_teste)
    detectados = 0
    
    for imagem_path in imagens_teste:
        resultados = modelo(imagem_path)
        for resultado in resultados:
            for caixa in resultado.boxes:
                classe = resultado.names[int(caixa.cls)]
                if classe in ["Knife", "Scissors", "Sword"]:
                    detectados += 1
                    break
    
    acuracia = (detectados / total) * 100 if total > 0 else 0
    print(f"Acurácia: {acuracia:.2f}%")

# ------------------- Pipeline Principal -------------------
def main():
    dataset = preparar_dataset()
    modelo = treinar_modelo()
    
    imagens_teste = [sample.filepath for sample in dataset]
    for imagem in imagens_teste:
        detectar_objetos(modelo, imagem)
    
    avaliar_modelo(modelo, dataset)

if __name__ == "__main__":
    main()
