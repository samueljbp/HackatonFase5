Detecção de Facas em Vídeos com YOLOv8
Este repositório contém um código Python para detecção de facas em vídeos, utilizando a biblioteca Ultralytics YOLOv8. O objetivo principal é identificar a presença de facas em vídeos, com o intuito de alertar sobre possíveis riscos de segurança.

Sumário
Introdução
Estrutura de Pastas
Configuração do Ambiente
Como Executar
Estrutura do Código
Dependências
Resultados
Próximos Passos
Licença

1. Introdução
   A detecção de objetos em vídeos é um problema comum em diversas áreas, como segurança, vigilância, análise de tráfego e muitas outras. A solução tradicional para este tipo de problema envolve o uso de algoritmos de visão computacional e aprendizado de máquina para identificar padrões e características que definem o objeto de interesse (neste caso, facas).

Este código utiliza o modelo YOLOv8 (You Only Look Once), um detector de objetos de última geração que se destaca pela sua velocidade e precisão. O YOLOv8 é capaz de identificar e localizar objetos em imagens e vídeos em tempo real, tornando-o ideal para aplicações que exigem alta performance.

2. Estrutura de Pastas
   A estrutura de pastas do projeto deve ser organizada da seguinte forma:

detecção-facas-yolov8/
├── data/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ └── val/
│ ├── images/
│ └── labels/
├── runs/
│ └── detect/
│ └── yolov8_knife_detection/
├── data.yaml
├── main.py
├── requirements.txt
└── secret.txt (não incluído no repositório)
Explicação:

data/: Contém o dataset de imagens e labels para treinamento e validação do modelo.
train/: Pasta para as imagens e labels de treinamento.
val/: Pasta para as imagens e labels de validação.
images/: Pasta para as imagens.
labels/: Pasta para os arquivos de label no formato YOLO.
runs/detect/: Pasta onde os resultados da detecção (vídeos e imagens com as detecções) são salvos.
data.yaml: Arquivo de configuração com informações sobre o dataset.
main.py: Script principal que coordena todas as etapas do processo.
requirements.txt: Arquivo com as dependências do projeto.
secret.txt: Arquivo (não incluído no repositório por questões de segurança) contendo a senha do e-mail remetente. 3. Configuração do Ambiente
Para executar o código, você precisará configurar o ambiente Python com as bibliotecas necessárias. Siga os seguintes passos:

Instale o Python: Se você ainda não tem o Python instalado, baixe a versão mais recente em https://www.python.org/downloads/ e siga as instruções de instalação.

Crie um ambiente virtual (opcional): É recomendado criar um ambiente virtual para isolar as dependências do projeto. Você pode fazer isso utilizando o módulo venv:

Bash

python3 -m venv .venv
Ative o ambiente virtual:

Linux/macOS:

<!-- end list -->

Bash

source .venv/bin/activate
Windows:

<!-- end list -->

Bash

.venv\Scripts\activate
Instale as dependências:

Bash

pip install -r requirements.txt
O arquivo requirements.txt contém todas as dependências do projeto.

4. Como Executar
   Clone o repositório:

Bash

git clone https://github.com/[seu-nome-de-usuario]/[nome-do-repositorio].git
Acesse o diretório do projeto:

Bash

cd detecção-facas-yolov8
Execute o script principal:

Bash

python main.py
O script main.py contém o código principal do sistema de detecção de facas.

5. Estrutura do Código
   O código é dividido em diversas etapas, desde a conversão de labels até o envio de e-mails com alertas. Abaixo, listamos os principais arquivos e suas funcionalidades:

main.py: Script principal que coordena todas as etapas do processo.
utils.py: Arquivo auxiliar com funções para conversão de labels, envio de e-mails e outras tarefas.
data.yaml: Arquivo de configuração com informações sobre o dataset.
secret.txt: Arquivo (não incluído no repositório por questões de segurança) contendo a senha do e-mail remetente. 6. Dependências
As principais dependências do projeto são:

ultralytics
Pillow
opencv-python
logging
random
smtplib
ssl
email
os
sys
Todas as dependências estão listadas no arquivo requirements.txt.

7. Resultados
   Os resultados da detecção de facas, incluindo as imagens e vídeos com as detecções, são salvos no diretório runs/detect.

8. Próximos Passos
   Melhorar a precisão do modelo: Coletar um dataset maior e mais diversificado de imagens de facas, ajustar os hiperparâmetros do modelo durante o treinamento e utilizar técnicas de aumento de dados para melhorar a precisão do modelo.
   Implementar um sistema de alerta em tempo real: Integrar o código com um sistema de câmeras de segurança para realizar a detecção em tempo real e enviar alertas imediatos em caso de detecção de facas.
   Implementar um painel de controle: Criar um painel de controle com interface gráfica para facilitar a visualização dos resultados, o gerenciamento dos alertas e a configuração do sistema.
9. Licença
   Este projeto está licenciado sob a licença MIT.

Contato
Em caso de dúvidas ou sugestões, entre em contato através do email [endereço de email removido].

Agradecimentos
Agradecemos à equipe do Ultralytics YOLOv8 pelo desenvolvimento da biblioteca e à comunidade de visão computacional pelo apoio e inspiração.
