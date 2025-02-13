import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import logging

# =========================
# 1. Configuração do Ambiente e Logs
# =========================

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# 2. Configuração do Sistema de Alerta (SMTP Seguro)
# =========================

EMAIL_ORIGEM = os.getenv('EMAIL_ORIGEM') # fakevisionguard@gmail.com
EMAIL_SENHA = os.getenv('EMAIL_SENHA') # wixb qyrv qvkc qqwh
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587  # Porta segura com TLS


def enviar_alerta(email_destino, mensagem):
    try:
        msg = MIMEText(mensagem)
        msg['Subject'] = '⚠️ Alerta de Objeto Cortante Detectado!'
        msg['From'] = EMAIL_ORIGEM
        msg['To'] = email_destino

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Criptografia TLS
            server.login(EMAIL_ORIGEM, EMAIL_SENHA)
            server.sendmail(EMAIL_ORIGEM, email_destino, msg.as_string())

        logging.info(f'Alerta enviado para {email_destino}')
    except Exception as e:
        logging.error(f'Erro ao enviar alerta: {e}')

# =========================
# 3. Carregar Modelo YOLO Treinado
# =========================

try:
    model = YOLO('yolov8n.pt')  # Caminho do modelo treinado
    logging.info('Modelo YOLO carregado com sucesso.')
except Exception as e:
    logging.error(f'Erro ao carregar o modelo YOLO: {e}')
    exit()

# =========================
# 4. Processamento de Vídeo
# =========================

video_files = ['video.mp4', 'video2.mp4']
resultados = []

for video_path in video_files:
    if not os.path.isfile(video_path):
        logging.error(f'Arquivo de vídeo não encontrado: {video_path}')
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f'Erro ao abrir o vídeo {video_path}')
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f'Analisando {video_path}: {frame_count} frames a {fps} FPS.')

    objetos_detectados = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning(f'Final do vídeo ou erro de leitura: {video_path}')
            break

        try:
            # Detecção de objetos com YOLO
            results = model(frame)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    if conf > 0.5:  # Confiança mínima
                        label = model.names[cls]
                        if label in ['knife', 'scissors', 'machete', 'cutter']:
                            # Desenhar caixa delimitadora
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                            # Enviar alerta por e-mail
                            email_destino = 'karinataccip@gmail.com'
                            enviar_alerta(email_destino, f'Alerta: {label} detectado com {conf*100:.1f}% de confiança!')

                            objetos_detectados += 1
                            logging.info(f'{label} detectado com {conf*100:.1f}% de confiança.')

        except Exception as e:
            logging.error(f'Erro no processamento do frame: {e}')

        cv2.imshow('VisionGuard - Detecção de Objetos Cortantes (YOLO)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info('Análise interrompida pelo usuário.')
            break

    cap.release()
    resultados.append(f'{video_path}: {objetos_detectados} objetos cortantes detectados.')

# =========================
# 5. Geração do Relatório Final
# =========================

with open('relatorio_detectados_yolo.txt', 'w') as f:
    f.write('Relatório de Detecção de Objetos Cortantes (YOLO):\n')
    f.write('\n'.join(resultados))

logging.info('Relatório gerado com sucesso em relatorio_detectados_yolo.txt')

# =========================
# 6. Encerramento
# =========================

cv2.destroyAllWindows()