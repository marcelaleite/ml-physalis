import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================

def get_kmers(sequence, k=4):
    """Transforma a sequência de nucleotídeos em palavras (k-mers) sobrepostas."""
    sequence = str(sequence).upper()
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_fasta_data(file_paths, label):
    """Lê arquivos FASTA e retorna um DataFrame com as sequências, IDs e rótulos."""
    data = []
    for file_path in file_paths:
        for record in SeqIO.parse(file_path, "fasta"):
            kmers = get_kmers(record.seq)
            data.append({"id": record.id, "sequence": kmers, "label": label})
    return pd.DataFrame(data)

# ==========================================
# 2. CARREGAMENTO DOS DADOS DE TREINO
# ==========================================
print("Carregando dados de treinamento...")

# Substitua pelos nomes corretos caso os arquivos estejam em outra pasta
df_pos = load_fasta_data(["p1.fasta", "p2.fasta"], label=1) # Classe Positiva
df_neg = load_fasta_data(["n1.fasta"], label=0)             # Classe Negativa

# Junta tudo e embaralha os dados
df_train = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# Extração de Features (Vetorização dos k-mers)
print("Vetorizando as sequências...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_train['sequence'])
y = df_train['label']

# Divisão em treino e teste (80% treino, 20% teste) para validação dos modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. TREINAMENTO E VALIDAÇÃO DOS MODELOS
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42) # probability=True é crucial para obter percentuais
}

best_model_name = None
best_model = None
best_f1 = 0

print("\n" + "="*40)
print("VALIDAÇÃO DOS MODELOS")
print("="*40)

for name, model in models.items():
    # Treinamento
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilidade da classe 1
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"--- {name} ---")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}\n")
    
    # Define o melhor modelo baseado no F1-Score (equilibra falsos positivos e falsos negativos)
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"🏆 Melhor modelo selecionado para a predição final: {best_model_name}")

# ==========================================
# 4. PREVISÃO NO ARQUIVO ALVO (Arabidopsis)
# ==========================================
print("\n" + "="*40)
print("PREVISÃO DE GENES DE RESISTÊNCIA NA ARABIDOPSIS")
print("="*40)

# Carrega a planta alvo. O label temporário é -1 (desconhecido)
df_target = load_fasta_data(["arabidopsis.fasta"], label=-1)

if df_target.empty:
    print("O arquivo arabidopsis.fasta está vazio ou não foi encontrado.")
else:
    # Transforma os k-mers da Arabidopsis usando o mesmo vetorizador do treino
    X_target = vectorizer.transform(df_target['sequence'])
    
    # Prevê as probabilidades
    target_probs = best_model.predict_proba(X_target)[:, 1]
    
    # Adiciona as probabilidades ao DataFrame
    df_target['probability'] = target_probs
    
    # Filtra apenas aqueles classificados como positivos (probabilidade > 50%)
    genes_encontrados = df_target[df_target['probability'] > 0.50].copy()
    
    # Ordena do mais provável para o menos provável
    genes_encontrados = genes_encontrados.sort_values(by='probability', ascending=False)
    
    if genes_encontrados.empty:
        print("Nenhum gene de resistência com probabilidade > 50% foi encontrado.")
    else:
        print(f"Foram encontrados {len(genes_encontrados)} possíveis genes de resistência:\n")
        print(f"{'ID DO GENE':<30} | {'PROBABILIDADE'}")
        print("-" * 50)
        for _, row in genes_encontrados.iterrows():
            gene_id = row['id'][:29] # Limita o tamanho do nome para não quebrar a tabela
            prob_percent = row['probability'] * 100
            print(f"{gene_id:<30} | {prob_percent:.2f}%")