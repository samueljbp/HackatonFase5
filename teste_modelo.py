from ultralytics import YOLO
import os  # Módulo para manipulação de diretórios e arquivos
import cv2
import smtplib
import mimetypes
from email.message import EmailMessage

# Configuração de diretórios
BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "resultados")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DETECTIONS_DIR = os.path.join(RESULTS_DIR, "detections")  # Pasta para salvar frames detectados

os.makedirs(DETECTIONS_DIR, exist_ok=True)  # Criar diretório se não existir

# Configuração do e-mail
EMAIL_SENDER = "fakevisionguard@gmail.com"
EMAIL_PASSWORD = "wixb qyrv qvkc qqwh"  # Idealmente, use variáveis de ambiente
EMAIL_RECIPIENT = "samuelandromeda@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# 1. Carregar o modelo treinado
model = YOLO("yolov8l.pt")

#print(model.names)

# 2. Especificar o caminho do vídeo
video_path = VIDEOS_DIR + "/video.mp4"  # Substitua pelo caminho do seu vídeo

target_classes = [43, 76]  # Substitua 47 pelo ID correto da classe faca no seu modelo

# 3. Processar o vídeo
""" results = model.predict(
    source=video_path,  # Caminho do vídeo
    conf=0.5,           # Limite de confiança para as detecções
    save=True,          # Salvar o vídeo com as detecções
    show=True,           # Exibir o vídeo em tempo real (opcional)
    classes=target_classes  # Filtrar as detecções por classes específicas
) """

# 4. Mensagem final
print("Processamento do vídeo concluído!")
#print(f"Vídeo com detecções salvo em: {results[0].save_dir}")

#exit(0)

# Enviar e-mail com o frame salvo
def send_email_with_attachment(to_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Detecta o tipo de arquivo
    mime_type, _ = mimetypes.guess_type(attachment_path)
    mime_main, mime_sub = mime_type.split("/")

    with open(attachment_path, "rb") as file:
        msg.add_attachment(file.read(), maintype=mime_main, subtype=mime_sub, filename=os.path.basename(attachment_path))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)


# Processar o vídeo frame a frame
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop quando o vídeo termina

    frame_count += 1
    results = model.predict(frame, conf=0.5, classes=target_classes)  # Realiza a detecção

    if results and results[0].boxes:  # Se houver detecções
        frame_filename = os.path.join(DETECTIONS_DIR, f"detection_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)  # Salva o frame com detecção
        
        send_email_with_attachment(
            EMAIL_RECIPIENT,
            "Alerta: Objeto Cortante Detectado",
            "Foi detectado um objeto cortante em um vídeo monitorado.",
            frame_filename
        )

        #pula 60 frames para evitar repetição
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 60)

        print(f"E-mail enviado com a imagem {frame_filename}")

cap.release()
print("Processamento do vídeo concluído!")