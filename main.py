import os  # Módulo para manipulação de diretórios e arquivos
import shutil  # Módulo para operações de cópia e remoção de arquivos
import smtplib  # Biblioteca para envio de e-mails
import ssl  # Biblioteca para criação de contexto seguro para envio de e-mails
import random  # Biblioteca para operações aleatórias
import torch  # Framework para deep learning
from email.message import EmailMessage  # Classe para criação de mensagens de e-mail
from ultralytics import YOLO  # YOLOv8 para detecção de objetos
import fiftyone as fo  # Biblioteca para manipulação de datasets de visão computacional
import fiftyone.zoo as foz  # Zoo de datasets do FiftyOne
from PIL import Image, ImageDraw  # Biblioteca para manipulação de imagens
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Métricas de avaliação
import cv2  # Biblioteca para manipulação de vídeos

# ------------------- Configurações -------------------
ADMIN_EMAIL = "admin@example.com"  # E-mail do administrador para envio de alertas
EMAIL_SENDER = "seu_email@example.com"  # E-mail que enviará as notificações
EMAIL_PASSWORD = "sua_senha"  # Senha do e-mail de envio (deve ser protegida)

# Diretórios base e específicos para armazenamento de imagens e rótulos
BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")

# Criação dos diretórios se não existirem
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "negativos"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "positivos_video"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "negativos_video"), exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(LABELS_TRAIN_DIR, exist_ok=True)
os.makedirs(LABELS_VAL_DIR, exist_ok=True)

# ------------------- Função de envio de e-mail -------------------
def enviar_email(imagem_path, objeto_detectado):
    """Envia um e-mail de alerta quando um objeto cortante é detectado."""
    assunto = "[Alerta] Objeto cortante detectado"
    corpo = f"Foi detectado um(a) {objeto_detectado} na imagem: {imagem_path}"

    msg = EmailMessage()
    msg.set_content(corpo)
    msg["Subject"] = assunto
    msg["From"] = EMAIL_SENDER
    msg["To"] = ADMIN_EMAIL

    # Criando contexto seguro para envio de e-mails
    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=contexto) as servidor:
        servidor.login(EMAIL_SENDER, EMAIL_PASSWORD)
        servidor.send_message(msg)
    print(f"E-mail enviado para {ADMIN_EMAIL} sobre a detecção de {objeto_detectado}.")

# ------------------- Limpeza dos diretórios -------------------
def limpar_pastas():
    """Remove e recria diretórios para garantir que a execução ocorra sem resíduos de execuções anteriores."""
    for pasta in [VAL_DIR, LABELS_VAL_DIR, RESULTS_DIR]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta, exist_ok=True)

# ------------------- Preparação do dataset -------------------
def preparar_dataset():
    """Baixa e organiza as imagens do dataset Open Images V7 para treinamento e validação."""
    limpar_pastas()
    classes_cortantes = ["Knife", "Scissors", "Sword"]  # Classes de objetos cortantes
    samples = []  # Lista para armazenar as amostras coletadas
    
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
        samples.extend(dataset)
    
    # Carregar imagens sem objetos cortantes para balancear o dataset
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

    random.shuffle(samples)  # Mistura as amostras para evitar viés

    for i, sample in enumerate(samples):
        destino_img = TRAIN_DIR if i < 0.8 * len(samples) else VAL_DIR
        destino_label = LABELS_TRAIN_DIR if i < 0.8 * len(samples) else LABELS_VAL_DIR
        shutil.copy(sample.filepath, destino_img)  # Copia imagem para diretório correspondente
        salvar_labels(sample, destino_label)  # Salva os rótulos das imagens

# ------------------- Salvar labels -------------------
def salvar_labels(sample, destino_label):
    """Salva os rótulos das imagens no formato YOLO."""
    label_path = os.path.join(destino_label, os.path.splitext(os.path.basename(sample.filepath))[0] + ".txt")
    with open(label_path, "w") as f:
        if sample.ground_truth and sample.ground_truth.detections:
            detections = sample.ground_truth.detections
            for det in detections:
                classe = det.label
                if classe in ["Knife", "Scissors", "Sword"]:
                    class_id = ["Knife", "Scissors", "Sword"].index(classe)
                    x, y, w, h = det.bounding_box  # YOLO usa bounding box normalizada
                    x_center = x + (w / 2)
                    y_center = y + (h / 2)
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# ------------------- Treinamento do modelo -------------------
def treinar_modelo():
    """Treina o modelo YOLOv8 para detecção de objetos cortantes."""
    modelo = YOLO("yolov8n.pt")
    modelo.train(data=os.path.join(DATASET_DIR, "data.yaml"), epochs=10, imgsz=640)
    return modelo

# ------------------- Avaliação do modelo -------------------
def avaliar_modelo(modelo):
    # Carregar imagens de validação
    imagens_val = [os.path.join(VAL_DIR, img) for img in os.listdir(VAL_DIR) if img.endswith('.jpg')]
    
    y_true, y_pred = [], []

    # Avaliar o modelo em cada imagem de validação
    for img_path in imagens_val:
        resultados = modelo(img_path)
        imagem = Image.open(img_path)
        draw = ImageDraw.Draw(imagem)
        pred_labels = []

        # Processa as previsões do modelo
        for resultado in resultados[0].boxes:
            classe_id = int(resultado.cls.item())
            # Verifica se a classe é um objeto cortante
            if classe_id < len(["Knife", "Scissors", "Sword"]):
                # Mapeia o ID da classe para o nome do objeto
                objeto_detectado = ["Knife", "Scissors", "Sword"][classe_id]

                # Adiciona a classe predita à lista de previsões
                pred_labels.append(classe_id)

                # Desenha a bounding box e o rótulo na imagem
                x1, y1, x2, y2 = resultado.xyxy[0].tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), objeto_detectado, fill="red")

                # Salva a imagem com as previsões
                imagem.save(os.path.join(RESULTS_DIR, os.path.basename(img_path)))

                # Enviar alerta se objeto cortante for identificado
                #enviar_email(img_path, objeto_detectado)
            # Salvar imagens negativas para análise
            else:
                pred_labels.append(0)

                # Salva a imagem negativa
                imagem.save(os.path.join(RESULTS_DIR, "negativos", os.path.basename(img_path)))

        # Carregar os labels reais da imagem
        label_file = os.path.join(LABELS_VAL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        true_labels = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                linhas = f.readlines()
                for linha in linhas:
                    true_labels.append(int(linha.split()[0]))
        else:
            true_labels.append(0)

        y_true.extend(true_labels)
        y_pred.extend(pred_labels)

    # Calcular métricas se houver dados
    if len(y_true) == len(y_pred) and len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Acurácia: {acc:.2f}, Precisão: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")
    else:
        print(f"Erro: Número de amostras em y_true ({len(y_true)}) e y_pred ({len(y_pred)}) não correspondem.")


# ------------------- Avaliação do modelo em vídeo -------------------
def avaliar_modelo_video(modelo, video_path):
    """Avalia o modelo em um vídeo, extraindo frames a cada segundo."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Obtém o FPS do vídeo
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Extrai um frame a cada segundo
        if frame_count % int(fps) == 0:
            # Converte o frame para o formato PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Realiza a detecção no frame
            resultados = modelo(frame_pil)
            draw = ImageDraw.Draw(frame_pil)
            pred_labels = []

            for resultado in resultados[0].boxes:
                classe_id = int(resultado.cls.item())
                if classe_id < len(["Knife", "Scissors", "Sword"]):
                    objeto_detectado = ["Knife", "Scissors", "Sword"][classe_id]
                    pred_labels.append(classe_id)
                    x1, y1, x2, y2 = resultado.xyxy[0].tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1), objeto_detectado, fill="red")

                    # Salva o frame com as previsões (opcional)
                    frame_pil.save(os.path.join(RESULTS_DIR, "positivos_video", f"frame_{frame_count // int(fps)}.jpg"))

                    # enviar_email(video_path, objeto_detectado)  # Adapte para enviar alertas de vídeo
                else:
                    # Salva o frame com as previsões (opcional)
                    frame_pil.save(os.path.join(RESULTS_DIR, "negativos_video", f"frame_{frame_count // int(fps)}.jpg"))
            

        frame_count += 1

    video.release()

# ------------------- Pipeline Principal -------------------
def main():
    preparar_dataset()
    modelo = treinar_modelo()
    avaliar_modelo(modelo)
    print("Treinamento e avaliação concluídos.")

    # Avalia o modelo no vídeo
    video_path = "videos/video.mp4"  # Substitua pelo caminho do seu vídeo
    avaliar_modelo_video(modelo, video_path)
    print("Avaliação em vídeo concluída.")

if __name__ == "__main__":
    main()
