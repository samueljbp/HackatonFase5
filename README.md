# Treinamento e Teste de Modelo YOLO

Este repositório contém notebooks para treinamento e teste de um modelo YOLO em imagens e vídeos.

## Requisitos

Antes de executar o notebook, instale as dependências necessárias:

```bash
pip install ultralytics opencv-python
```

Além disso, é necessário colocar as imagens e labels de treinamento nos seguintes caminhos do seu Google Drive:

Imagens de treino:
"/content/drive/MyDrive/Hackaton/Dataset/val/images"

Labels de treino:
"/content/drive/MyDrive/Hackaton/Dataset/val/labels"

Para o notebook de teste com o video, coloque o arquivo do modelo yolo11m.pt no seguinte caminho do ambiente colab:
/content/yolo11m.pt

E coloque o arquivo video.mp4 no seguinte caminho do ambiente colab:
/content/video.mp4

## Estrutura do Notebook

1. **Conversão das Labels**: Preparação e organização dos dados rotulados.
2. **Testes em imagens**: Avaliação do modelo YOLO em um conjunto de imagens.
3. **Testes em vídeos**: Aplicação do modelo YOLO em vídeos para detecção de objetos em movimento.

## Como Usar

1. **Executar o notebook**: Carregue e execute cada célula do notebook para realizar o treinamento e os testes.
2. **Adicionar novos dados**: Caso queira testar com novas imagens ou vídeos, certifique-se de adicionar os arquivos nas pastas correspondentes.

## Exemplo de Uso

Para testar o modelo em uma imagem, utilize:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("caminho/para/imagem.jpg")
results.show()
```

Para testar em um vídeo:

```python
results = model("caminho/para/video.mp4")
results.show()
```

## Exemplo de Uso

Para testar a detecção em um vídeo:

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("caminho/para/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    results.show()
```

## Contribuição

Sinta-se à vontade para contribuir com melhorias no notebook, como otimização do modelo ou adição de novos datasets.

## Licença

Este projeto está sob a licença MIT.
