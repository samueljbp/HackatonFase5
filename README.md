# README - Detecção de Objetos Cortantes com YOLO

Este projeto utiliza o modelo YOLO para detectar objetos cortantes (facas, tesouras, espadas, etc.) em imagens e vídeos. O sistema também pode enviar alertas por e-mail ao identificar um objeto cortante.

## 1. Instalação de Dependências

Antes de executar o código, instale todas as dependências necessárias:

```bash
pip install ultralytics fiftyone opencv-python pillow smtplib
```

### Explicação das Dependências

-   `ultralytics`: Biblioteca que fornece implementações otimizadas do modelo YOLO para detecção de objetos.
-   `fiftyone`: Usada para gerenciar e visualizar datasets anotados, facilitando o trabalho com imagens rotuladas.
-   `opencv-python`: Biblioteca de processamento de imagens e vídeos, utilizada para capturar e exibir os resultados do modelo.
-   `pillow`: Biblioteca para manipulação de imagens, como carregamento e conversão de formatos.
-   `smtplib`: Módulo padrão do Python usado para enviar e-mails, essencial para a funcionalidade de alertas.

## 2. Estrutura de Diretórios Necessária

Certifique-se de que a seguinte estrutura de diretórios está configurada corretamente:

```
Hackaton/
│── Dataset/
│   │── train/
│   │   │── images/  # Contém imagens de treino
│   │   │── labels/  # Contém labels no formato YOLO
│   │── val/
│   │   │── images/  # Contém imagens de validação
│   │   │── labels/  # Contém labels no formato YOLO
│── models/
│   │── best.pt  # Modelo treinado salvo aqui
│── videos/
│   │── test_video.mp4  # Vídeos para teste do modelo
```

## 3. Treinamento do Modelo

Para treinar o modelo, utilize o seguinte comando:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Escolha um modelo base
model.train(data="Hackaton/Dataset/dataset.yaml", epochs=50)
```

O modelo treinado será salvo na pasta `models/`.

## 4. Testando o Modelo com um Vídeo

Para testar o modelo em um vídeo, utilize:

```python
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
cap = cv2.VideoCapture("videos/test_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    cv2.imshow("YOLO Detection", results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

## 5. Envio de Alertas por E-mail

O envio de alertas por email será feito automaticamente ao executar o trecho abaixo:

# Enviando os e-mails

sender_email = REMETENTE
for receiver_email in DESTINATARIOS_VALIDOS:
enviar_email(sender_email, receiver_email, QTD, VIDEO, PASSWORD)

Este trecho funcionará automaticamente desde que os trechos anteriores tenham sido executados. As variáveis já estarão todas preenchidas.

## 6. Considerações Finais

-   Certifique-se de ter permissões para acessar os diretórios corretos.
-   O modelo deve ser treinado antes de realizar testes.
-   Para envio de e-mails, pode ser necessário ativar "Acesso a aplicativos menos seguros" no provedor de e-mail ou configurar autenticação por aplicativo.

Caso tenha dúvidas, consulte a documentação do YOLO e das bibliotecas utilizadas.
