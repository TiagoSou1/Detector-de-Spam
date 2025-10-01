# --- Conteúdo para o arquivo src/main.py ---

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(">>> INICIANDO SCRIPT DE DETECÇÃO DE SPAM <<<")

# --- Passo 0: Configuração Inicial do NLTK ---
# Tenta carregar as stopwords. Se não conseguir, faz o download.
try:
    stopwords.words('english')
except LookupError:
    print("\n[INFO] Baixando recursos do NLTK (stopwords)...")
    nltk.download('stopwords')
    print("[INFO] Download concluído.")

# --- Passo 1: Carregar o Dataset ---
# O caminho '../data/spam.csv' sobe um nível a partir de 'src' e entra em 'data'
caminho_arquivo = 'data\spam.csv'
print(f"\n[PASSO 1] Carregando o dataset de '{caminho_arquivo}'...")

try:
    # O encoding 'latin-1' é necessário para este arquivo específico
    df = pd.read_csv(caminho_arquivo, encoding='latin-1')
    # Manter apenas as colunas necessárias e renomeá-las
    df = df[['v1', 'v2']]
    df.columns = ['categoria', 'mensagem']
    print(f"[INFO] Dataset carregado com sucesso! Shape: {df.shape}")
except FileNotFoundError:
    print(f"\n[ERRO] Arquivo não encontrado em '{caminho_arquivo}'.")
    print("Verifique se o arquivo spam.csv está na pasta 'data'.")
    exit()
except Exception as e:
    print(f"[ERRO] Ocorreu um erro ao carregar o dataset: {e}")
    exit()

# --- Passo 2: Pré-processamento dos Textos ---
print("\n[PASSO 2] Limpando e pré-processando os textos...")

stop_words = set(stopwords.words('english'))

def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^a-z\s]', '', texto)
    palavras = texto.split()
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]
    return " ".join(palavras_filtradas)

df['mensagem_limpa'] = df['mensagem'].apply(preprocessar_texto)
df['categoria'] = df['categoria'].map({'spam': 1, 'ham': 0})
print("[INFO] Pré-processamento concluído.")

# --- Passo 3: Vetorização dos Textos com TF-IDF ---
print("\n[PASSO 3] Convertendo textos em vetores numéricos...")

X = df['mensagem_limpa']
y = df['categoria']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("[INFO] Textos vetorizados com sucesso.")

# --- Passo 4: Treinamento do Modelo Naive Bayes ---
print("\n[PASSO 4] Treinando o modelo de classificação...")

modelo = MultinomialNB()
modelo.fit(X_train_tfidf, y_train)
print("[INFO] Modelo treinado com sucesso!")

# --- Passo 5: Avaliação do Modelo ---
print("\n[PASSO 5] Avaliando o desempenho do modelo...")

y_pred = modelo.predict(X_test_tfidf)
acuracia = accuracy_score(y_test, y_pred)
print(f"\n> Acurácia: {acuracia:.4f}")
print("\n> Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Ham (Não Spam)', 'Spam']))

print("\n> Gerando Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham (Não Spam)', 'Spam'],
            yticklabels=['Ham (Não Spam)', 'Spam'])
plt.xlabel('Rótulo Previsto')
plt.ylabel('Rótulo Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# --- Passo 6: Teste com Novas Mensagens ---
print("\n[PASSO 6] Testando o modelo com novas mensagens do mundo real...")

def classificar_mensagem(mensagem):
    mensagem_limpa = preprocessar_texto(mensagem)
    mensagem_vetorizada = vectorizer.transform([mensagem_limpa])
    predicao = modelo.predict(mensagem_vetorizada)
    probabilidade = modelo.predict_proba(mensagem_vetorizada)
    
    if predicao[0] == 1:
        return f"É SPAM! (Confiança: {probabilidade[0][1]:.2%})"
    else:
        return f"Não é Spam (Ham). (Confiança: {probabilidade[0][0]:.2%})"

mensagem_spam = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461."
mensagem_ham = "Sorry, I'll call later. I'm in a meeting right now."

print(f"\n- Mensagem: '{mensagem_spam}'")
print(f"  Classificação: {classificar_mensagem(mensagem_spam)}")
print(f"\n- Mensagem: '{mensagem_ham}'")
print(f"  Classificação: {classificar_mensagem(mensagem_ham)}")

print("\n>>> SCRIPT CONCLUÍDO <<<")