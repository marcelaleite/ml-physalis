import os
import pandas as pd
import numpy as np
from Bio import SeqIO, Entrez
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# 0. CONFIGURAÇÃO DO NCBI E DOWNLOAD
# ==========================================
# ATENÇÃO: O NCBI exige um e-mail válido para identificar as requisições.
Entrez.email = "seu_email@provedor.com" 

# Pasta onde os arquivos salvos ficarão centralizados
OUTPUT_FOLDER = "dados_proteinas"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mapeamento de arquivos para os IDs de Acesso (Accession IDs) reais do NCBI Protein.
# SUBSTITUA ESTES EXEMPLOS PELOS IDS REAIS DO SEU PROJETO SE NECESSÁRIO.
ncbi_accessions = {
    "p1.fasta": ["NP_001322204.1"],  # Ex: Proteína de resistência LOV1 (Arabidopsis)
    "p2.fasta": ["NP_001319970.1"],  # Ex: Proteína de resistência SNC1 (Arabidopsis)
    "n1.fasta": ["NP_177114.1"],      # Ex: Actina-7 (Usada frequentemente como controle negativo)
    
    # Lista de IDs de proteínas da planta alvo que você quer testar/prever
    "arabidopsis.fasta": ["NP_172561.2", "NP_001117266.1", "NP_195400.1"] 
}

def download_fasta_from_ncbi(accession_ids, filename):
    """Conecta ao NCBI, baixa as sequências de proteínas e salva localmente."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    print(f"📥 Baixando {filename} do NCBI (IDs: {accession_ids})...")
    try:
        # efetch busca diretamente no banco de dados de proteínas ('protein')
        handle = Entrez.efetch(db="protein", id=accession_ids, rettype="fasta", retmode="text")
        data = handle.read()
        handle.close()
        
        with open(file_path, "w") as f:
            f.write(data)
        print(f"✅ Salvo com sucesso em: {file_path}")
    except Exception as e:
        print(f"❌ Erro ao baixar o arquivo {filename}: {e}")

# Executa o download automático de todas as classes configuradas
for fname, ids in ncbi_accessions.items():
    download_fasta_from_ncbi(ids, fname)

# ==========================================
# 1. FUNÇÕES DE PROCESSAMENTO DE AMINOÁCIDOS
# ==========================================

def get_kmers(sequence, k=3):
    """Transforma a sequência de aminoácidos em tripeptídeos sobrepostos."""
    sequence = str(sequence).upper()
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_fasta_data(filename, label):
    """Lê os arquivos FASTA baixados e monta o DataFrame inicial."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    data = []
    if not os.path.exists(file_path):
        return pd.DataFrame(data)
        
    for record in SeqIO.parse(file_path, "fasta"):
        kmers = get_kmers(record.seq)
        data.append({"id": record.id, "sequence": kmers, "label": label})
    return pd.DataFrame(data)

# ==========================================
# 2. PREPARAÇÃO DOS DADOS DE TREINAMENTO
# ==========================================
print("\nProcessando arquivos de aminoácidos...")

df_pos = pd.concat([load_fasta_data("p1.fasta", label=1), load_fasta_data("p2.fasta", label=1)])
df_neg = load_fasta_data("n1.fasta", label=0)

if df_pos.empty or df_neg.empty:
    print("❌ Erro crítico: O DataFrame de treino está vazio. Verifique suas conexões ou IDs do NCBI.")
    exit()

# Combina e embaralha as amostras
df_train = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# Vetorização (Extração estatística de k-mers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_train['sequence'])
y = df_train['label']

# Split estratificado para validação (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. TREINAMENTO E VALIDAÇÃO DOS MODELOS
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

best_model_name = None
best_model = None
best_f1 = 0

print("\n" + "="*50)
print("             VALIDAÇÃO DOS MODELOS            ")
print("="*50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Cálculo das métricas solicitadas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    
    print(f"--- {name} ---")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}\n")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"🏆 Modelo escolhido para a classificação final: {best_model_name}")

# ==========================================
# 4. PREDIÇÃO NA PLANTA ALVO (ARABIDOPSIS)
# ==========================================
print("\n" + "="*50)
print("     PREDIÇÃO DE PROTEÍNAS DE RESISTÊNCIA     ")
print("="*50)

df_target = load_fasta_data("arabidopsis.fasta", label=-1)

if df_target.empty:
    print("⚠️ O arquivo arabidopsis.fasta não possui sequências válidas para predição.")
else:
    # Transforma o alvo usando o vocabulário construído no treino
    X_target = vectorizer.transform(df_target['sequence'])
    
    # Prediz as probabilidades com o melhor modelo validado
    target_probs = best_model.predict_proba(X_target)[:, 1]
    df_target['probability'] = target_probs
    
    # Filtra e ordena as proteínas com probabilidade maior que 50%
    predicoes = df_target[df_target['probability'] > 0.50].sort_values(by='probability', ascending=False)
    
    if predicoes.empty:
        print("Nenhuma sequência candidata a proteína de resistência ultrapassou o limiar de 50%.")
    else:
        print(f"Foram encontradas {len(predicoes)} possíveis proteínas de resistência:\n")
        print(f"{'ID DA PROTEÍNA (NCBI)':<35} | {'PROBABILIDADE'}")
        print("-" * 55)
        for _, row in predicoes.iterrows():
            print(f"{row['id'][:34]:<35} | {row['probability']*100:.2f}%")