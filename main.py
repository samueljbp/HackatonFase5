import os
import shutil
import smtplib
import ssl
import random
import torch
from email.message import EmailMessage
from ultralytics import YOLO  # YOLOv8 para detecção de objetos
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    for pasta in [VAL_DIR, LABELS_VAL_DIR, RESULTS_DIR]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta, exist_ok=True)

# ------------------- Preparação do dataset -------------------
def preparar_dataset():
    limpar_pastas()
    classes_cortantes = ["Knife", "Scissors", "Sword"]
    samples = []
    
    for classe in classes_cortantes:
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
        print(f"Carregando dataset de {classe}", len(dataset))
        samples.extend(dataset)  # Adicionando amostras ao conjunto geral
    
    # Carregar imagens sem objetos cortantes
    dataset_sem_cortantes = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        exclude_classes=classes_cortantes,
        max_samples=200,
        shuffle=True,
        cleanup=True,
        dataset_name="open-images-v7-no-cutting-tools"
    )
    print(f"Carregando dataset sem objetos cortantes", len(dataset_sem_cortantes))
    samples.extend(dataset_sem_cortantes)

    random.shuffle(samples)
    
    for i, sample in enumerate(samples):
        destino_img = TRAIN_DIR if i < 0.8 * len(samples) else VAL_DIR
        destino_label = LABELS_TRAIN_DIR if i < 0.8 * len(samples) else LABELS_VAL_DIR
        shutil.copy(sample.filepath, destino_img)
        salvar_labels(sample, destino_label)

# ------------------- Salvar labels -------------------
def salvar_labels(sample, destino_label):
    label_path = os.path.join(destino_label, os.path.splitext(os.path.basename(sample.filepath))[0] + ".txt")
    with open(label_path, "w") as f:
        if sample.ground_truth and sample.ground_truth.detections:
            detections = sample.ground_truth.detections
            for det in detections:
                classe = det.label
                if classe in ["Knife", "Scissors", "Sword"]:
                    class_id = ["Knife", "Scissors", "Sword"].index(classe)
                    x, y, w, h = det.bounding_box
                    x_center = x + (w / 2)
                    y_center = y + (h / 2)
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# ------------------- Treinamento do modelo -------------------
def treinar_modelo():
    modelo = YOLO("yolov8n.pt")
    modelo.train(data=os.path.join(DATASET_DIR, "data.yaml"), epochs=15, imgsz=640)
    return modelo

# ------------------- Avaliação do modelo -------------------
def avaliar_modelo(modelo):
    imagens_val = [os.path.join(VAL_DIR, img) for img in os.listdir(VAL_DIR) if img.endswith('.jpg')]
    y_true, y_pred = [], []

    for img_path in imagens_val:
        resultados = modelo(img_path)
        imagem = Image.open(img_path)
        draw = ImageDraw.Draw(imagem)

        for resultado in resultados[0].boxes:
            classe_id = int(resultado.cls.item())
            objeto_detectado = ["Knife", "Scissors", "Sword"][classe_id]
            y_pred.append(classe_id)

            x1, y1, x2, y2 = resultado.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), objeto_detectado, fill="red")
            imagem.save(os.path.join(RESULTS_DIR, os.path.basename(img_path)))

            #enviar_email(img_path, objeto_detectado)

        label_file = os.path.join(LABELS_VAL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                linhas = f.readlines()
                for linha in linhas:
                    y_true.append(int(linha.split()[0]))

    if y_true and y_pred:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Acurácia: {acc:.2f}, Precisão: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")

# ------------------- Pipeline Principal -------------------
def main():
    preparar_dataset()
    modelo = treinar_modelo()
    avaliar_modelo(modelo)
    print("Treinamento e avaliação concluídos.")

if __name__ == "__main__":
    main()