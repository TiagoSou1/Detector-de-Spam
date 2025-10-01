
# 📨 Detecção de Spam com Python e Naive Bayes

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Acurácia](https://img.shields.io/badge/Acurácia-0.98-green)](#modelo-de-classificação)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Descrição do Projeto

Este projeto implementa um **sistema de detecção de spam** em mensagens de texto utilizando Python. O modelo é treinado com o dataset **spam.csv**, contendo mensagens classificadas como `spam` ou `ham` (não spam). Utilizamos técnicas de:

- Pré-processamento de texto
- Vetorização TF-IDF
- Classificação com **Multinomial Naive Bayes**

O objetivo é identificar automaticamente mensagens de spam com alta precisão e permitir testes com novas mensagens do mundo real.

---

## Estrutura do Projeto



DETECAO_DE_SPAM/
├── data/
│   └── spam.csv            # Dataset com mensagens de spam/ham
├── src/
│   └── main.py             # Script principal
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação



---

## Tecnologias e Bibliotecas

- Python 3.10+
- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn
- re (expressões regulares)

---

## Como Rodar

1. Clonar o repositório:

git clone https://github.com/TiagoSou1/deteccao_de_spam.git
cd DETECAO_DE_SPAM


2. Instalar dependências:

pip install -r requirements.txt


3. Executar o script:

python src/main.py

O script realiza:

* Carregamento do dataset `spam.csv`
* Pré-processamento das mensagens
* Vetorização TF-IDF
* Treinamento do modelo Multinomial Naive Bayes
* Avaliação do modelo com **acurácia**, **relatório de classificação** e **matriz de confusão**
* Testes com novas mensagens

---

## Pré-processamento

* Conversão para minúsculas
* Remoção de URLs e caracteres especiais
* Remoção de **stopwords** em inglês
* Tokenização simples e reconstrução do texto limpo

---

## Dataset

O arquivo `spam.csv` deve conter:

| Coluna | Descrição                   |
| ------ | --------------------------- |
| v1     | Categoria (`spam` ou `ham`) |
| v2     | Mensagem de texto           |

O script renomeia para:

* `categoria` → 1 = Spam, 0 = Ham
* `mensagem` → Texto original
* `mensagem_limpa` → Texto pré-processado

---

## Modelo de Classificação

* **Algoritmo:** Multinomial Naive Bayes
* **Vetorização:** TF-IDF (`max_features=3000`)
* **Divisão de dados:** 80% treino / 20% teste
* **Métricas:** Acurácia, Precision, Recall, F1-score

> Exemplo de saída:

```text
> Mensagem: "Congratulations! You've won a $1000 gift card!"
> Classificação: É SPAM! (Confiança: 95.23%)
```

---

## Visualizações

O script gera uma **matriz de confusão** com `seaborn` para análise do desempenho do modelo.

---

## Próximos Passos / Melhorias

* Testar outros modelos (Random Forest, Logistic Regression)
* Implementar **stemming** ou **lemmatization**
* Criar **interface web/API** para uso em tempo real
* Explorar modelos de **Deep Learning** (RNN, LSTM) para detecção de spam

---

## Licença

Este projeto está licenciado sob a licença MIT.
