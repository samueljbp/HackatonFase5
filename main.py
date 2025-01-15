import os
import shutil
import smtplib
import ssl
import random
from email.message import EmailMessage
from ultralytics import YOLO  # YOLOv8 para detecção de objetos
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image, ImageDraw

# ------------------- Configurações -------------------
ADMIN_EMAIL = "admin@example.com"  # Parametrize o e-mail do administrador
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

# ------------------- Limpeza dos diretórios -------------------
def limpar_pastas():
    for pasta in [TRAIN_DIR, VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR, RESULTS_DIR]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta, exist_ok=True)

# ------------------- Avaliação do modelo -------------------
def avaliar_modelo(modelo):
    imagens_val = [os.path.join(VAL_DIR, img) for img in os.listdir(VAL_DIR) if img.endswith('.jpg')]
    for img_path in imagens_val:
        resultados = modelo(img_path)
        for resultado in resultados[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, classe_id = resultado
            objeto_detectado = ["Knife", "Scissors", "Sword"][int(classe_id)]

            # Marcar a imagem
            imagem = Image.open(img_path)
            draw = ImageDraw.Draw(imagem)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), objeto_detectado, fill="red")
            imagem.save(os.path.join(RESULTS_DIR, os.path.basename(img_path)))

            # Enviar e-mail
            #enviar_email(img_path, objeto_detectado)

# ------------------- Download e preparação do dataset -------------------
def preparar_dataset():
    limpar_pastas()
    class_map = {"Knife": 0, "Scissors": 1, "Sword": 2}
    datasets = []

    # Baixar imagens com objetos cortantes
    for classe in class_map.keys():
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[classe],
            max_samples=200,
            shuffle=True,
            cleanup=True,
            dataset_name=f"open-images-v7-{classe}-validation"
        )
        datasets.append(dataset)

    # Combina todos os datasets
    dataset_combined = fo.Dataset(name="objetos_cortantes")
    for dataset in datasets:
        dataset_combined.add_samples(dataset)
    
    # Baixar imagens sem objetos cortantes
    dataset_negativo = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=[],
        max_samples=200,
        shuffle=True,
        cleanup=True,
        dataset_name="open-images-v7-negativo"
    )
    for sample in dataset_negativo:
        dataset_combined.add_sample(sample)

    dataset_combined.shuffle()

    for i, sample in enumerate(dataset_combined):
        destino_img = TRAIN_DIR if i % 5 != 0 else VAL_DIR
        shutil.copy2(sample.filepath, destino_img)

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

# ------------------- Pipeline Principal -------------------
def main():
    preparar_dataset()
    modelo = treinar_modelo()
    avaliar_modelo(modelo)
    print("Treinamento e avaliação concluídos.")

if __name__ == "__main__":
    main()
