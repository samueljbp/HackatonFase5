# Detecção de Facas em Vídeos com YOLOv8

Este repositório contém um código Python para detecção de facas em vídeos, utilizando a biblioteca Ultralytics YOLOv8. O objetivo principal é identificar a presença de facas em vídeos, com o intuito de alertar sobre possíveis riscos de segurança.

## Sumário
- [Introdução](#introducao)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Configuração do Ambiente](#configuracao-do-ambiente)
- [Como Executar](#como-executar)
- [Estrutura do Código](#estrutura-do-codigo)
- [Dependências](#dependencias)
- [Resultados](#resultados)
- [Próximos Passos](#proximos-passos)
- [Licença](#licenca)

## 1. Introdução <a name="introducao"></a>

A detecção de objetos em vídeos é um problema comum em diversas áreas, como segurança, vigilância, análise de trâfego e muitas outras. A solução tradicional para este tipo de problema envolve o uso de algoritmos de visão computacional e aprendizado de máquina para identificar padrões e características que definem o objeto de interesse (neste caso, facas).

Este código utiliza o modelo **YOLOv8** (*You Only Look Once*), um detector de objetos de última geração que se destaca pela sua velocidade e precisão. O YOLOv8 é capaz de identificar e localizar objetos em imagens e vídeos em tempo real, tornando-o ideal para aplicações que exigem alta performance.

## 2. Estrutura de Pastas <a name="estrutura-de-pastas"></a>

A estrutura de pastas do projeto deve ser organizada da seguinte forma:

```bash
detecção-facas-yolov8/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── runs/
│   └── detect/
│       └── yolov8_knife_detection/
├── data.yaml
├── main.py
├── requirements.txt
└── secret.txt (não incluído no repositório)
```

### Explicação:
- `data/`: Contém o dataset de imagens e labels para treinamento e validação do modelo.
- `train/`: Pasta para as imagens e labels de treinamento.
- `val/`: Pasta para as imagens e labels de validação.
- `runs/detect/`: Pasta onde os resultados da detecção (vídeos e imagens com as detecções) são salvos.
- `data.yaml`: Arquivo de configuração com informações sobre o dataset.
- `main.py`: Script principal que coordena todas as etapas do processo.
- `requirements.txt`: Arquivo com as dependências do projeto.
- `secret.txt`: Arquivo (não incluído no repositório por questões de segurança) contendo a senha do e-mail remetente.

## 3. Configuração do Ambiente <a name="configuracao-do-ambiente"></a>

Para executar o código, você precisará configurar o ambiente Python com as bibliotecas necessárias. Siga os seguintes passos:

1. **Instale o Python**: Baixe a versão mais recente em [python.org](https://www.python.org/downloads/).
2. **Crie um ambiente virtual (opcional)**:
   ```bash
   python3 -m venv .venv
   ```
3. **Ative o ambiente virtual**:
   - **Linux/macOS**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
4. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Como Executar <a name="como-executar"></a>

1. Clone o repositório:
   ```bash
   git clone https://github.com/[seu-usuario]/[nome-do-repositorio].git
   ```
2. Acesse o diretório do projeto:
   ```bash
   cd detecção-facas-yolov8
   ```
3. Execute o script principal:
   ```bash
   python main.py
   ```

## 5. Estrutura do Código <a name="estrutura-do-codigo"></a>

- `main.py`: Script principal que coordena todas as etapas do processo.
- `utils.py`: Arquivo auxiliar com funções para conversão de labels, envio de e-mails e outras tarefas.
- `data.yaml`: Arquivo de configuração com informações sobre o dataset.

## 6. Dependências <a name="dependencias"></a>

As principais dependências do projeto são:

- `ultralytics`
- `Pillow`
- `opencv-python`
- `logging`
- `random`
- `smtplib`
- `ssl`
- `email`
- `os`
- `sys`

Todas as dependências estão listadas no arquivo `requirements.txt`.

## 7. Resultados <a name="resultados"></a>

Os resultados da detecção de facas, incluindo as imagens e vídeos com as detecções, são salvos no diretório `runs/detect`.

## 8. Próximos Passos <a name="proximos-passos"></a>

- **Melhorar a precisão do modelo**: Ajuste de hiperparâmetros e aumento do dataset.
- **Implementar um sistema de alerta em tempo real**: Integração com câmeras de segurança.
- **Criar um painel de controle**: Interface gráfica para facilitar a visualização dos resultados.

## 9. Licença <a name="licenca"></a>

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Contato

Em caso de dúvidas ou sugestões, entre em contato através do email [seu-email@example.com](mailto:seu-email@example.com).

## Agradecimentos

Agradecemos à equipe do **Ultralytics YOLOv8** pelo desenvolvimento da biblioteca e à comunidade de visão computacional pelo apoio e inspiração.

