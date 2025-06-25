# Projeto Alzheimer - Classificação de Imagens

Este projeto realiza a classificação de imagens para detecção de Alzheimer utilizando redes neurais convolucionais. O pipeline já inclui todo o pré-processamento necessário para as imagens.

## Requisitos

- **Python 3.10.11** (recomendada exatamente esta versão para evitar incompatibilidades)
- TensorFlow (veja observação abaixo)

## Como usar o projeto

### 1. Crie um ambiente virtual

No terminal, execute:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instale as dependências

Com o ambiente virtual ativado, rode:

```bash
pip install -r requirements.txt
```

> **Observação sobre o TensorFlow:**  
> No arquivo `requirements.txt` está especificada a versão base do TensorFlow.  
> Dependendo do seu sistema operacional, você pode instalar uma versão otimizada:
> - **Mac com Apple Silicon:**  
>   ```bash
>   pip install tensorflow-macos tensorflow-metal
>   ```
> - **Linux/Windows com GPU NVIDIA:**  
>   Siga as instruções oficiais do [TensorFlow](https://www.tensorflow.org/install) para instalar a versão com suporte a GPU.
> 
> Se preferir, mantenha a versão base do requirements, que funciona em qualquer sistema, mas pode ser mais lenta.

### 3. Prepare os dados

- Extraia o arquivo `Data.zip` para que a pasta `Data` fique no diretório do projeto.
- As imagens já estão pré-processadas e prontas para uso.

### 4. Execute o notebook ou arquivo .py
Abra o arquivo `alzheimer.py` e execute para treinar e avaliar o modelo da arquitetura 1.
Abra o arquivo `alzheimer.ipynb` e execute as células para treinar e avaliar o modelo da arquitetura 2.

---

## Caso queira usar imagens baixadas do Kaggle

1. Baixe o dataset do Kaggle e coloque a pasta dentro do diretório `Data`.
2. Execute o notebook `treat_images.ipynb` para realizar todo o pré-processamento e balanceamento das imagens.
3. Depois, siga normalmente para o passo 4 acima.

---

## Estrutura do projeto

- `alzheimer.ipynb` — Notebook principal de treinamento e avaliação apresentado como **Arquitetura 2**.
- `alzheimer.py` — Script de treinamento e avaliação apresentado como **Arquitetura 1**.
- `treat_images.ipynb` — Notebook de pré-processamento e balanceamento (use apenas se for baixar dados brutos).
- `requirements.txt` — Lista de dependências.
- `Data/` — Pasta com as imagens já pré-processadas.