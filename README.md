# Predição de Risco de AVC (Stroke Risk Prediction)

Projeto prático interdisciplinar de Ciência de Dados, Engenharia de Dados e Análise de Dados desenvolvido para a disciplina de **Métodos Quantitativos (TADS 2026.1)**. O sistema cobre o ciclo de vida completo de um modelo de Machine Learning: desde a exploração inicial dos dados brutos e tratamento de imperfeições, passando pelo pipeline ETL automatizado e treinamento com balanceamento, até a disponibilização final em um painel interativo **Streamlit**.

---

## Componentes do Grupo

- **[Insira o Nome do Aluno 1]** — *(Matrícula / Papel no Projeto)*
- **[Insira o Nome do Aluno 2]** — *(Matrícula / Papel no Projeto)*
- **[Insira o Nome do Aluno 3]** — *(Matrícula / Papel no Projeto)*
- **[Insira o Nome do Aluno 4]** — *(Matrícula / Papel no Projeto)*

*(Substitua os campos acima pelos nomes e dados reais dos integrantes do grupo).*

---

## Stack Tecnológica Utilizada

O projeto foi construído sobre um ecossistema moderno e modular em **Python**:

- **Core & Manipulação de Dados (Fase 1 e 2):**
  - [`Python 3`](https://www.python.org/)
  - [`Pandas`](https://pandas.pydata.org/) & [`NumPy`](https://numpy.org/) — manipulação vetorial, criação de faixas (*binning*) e limpeza de dados.
- **Visualização & BI (Fase 1 e 4):**
  - [`Plotly Express` & `Plotly Graph Objects`](https://plotly.com/python/) — criação de gráficos interativos, medidores dinâmicos de probabilidade (*Gauge Chart*), curvas ROC e matrizes de confusão.
  - [`Seaborn`](https://seaborn.pydata.org/) & [`Matplotlib`](https://matplotlib.org/) — visualizações exploratórias no notebook de EDA.
- **Engenharia de Features & Machine Learning (Fase 2 e 3):**
  - [`Scikit-Learn`](https://scikit-learn.org/) — `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder` e `LogisticRegression(class_weight="balanced")`.
  - [`Imbalanced-Learn (imblearn)`](https://imbalanced-learn.org/) — aplicação da técnica de sobreamostragem **SMOTE** para balancear a classe minoritária de AVC.
  - [`Joblib`](https://joblib.readthedocs.io/) — serialização e exportação do pré-processador (`preprocessor.pkl`) e do modelo treinado (`stroke_risk_model.joblib`).
- **Aplicação Web Interativa (Fase 5):**
  - [`Streamlit`](https://streamlit.io/) — framework de interface web em Python, customizado com **CSS Adaptativo (Light/Dark Mode)** na paleta do **Claude Code** para experiência visual sóbria e profissional.

---

## Estrutura de Diretórios

```text
stroke-risk-prediction/
│
├── app/
│   ├── templates/          # Arquivos de suporte a templates
│   └── app.py              # Aplicação Web Streamlit (Fase 5)
│
├── data/
│   ├── raw/                # Dataset bruto (healthcare-dataset-stroke-data.csv)
│   └── processed/          # CSVs processados e codificados, com a distribuição real das classes (`preprocessor.pkl`)
│
├── models/
│   └── stroke_risk_model.joblib # Artefato contendo o modelo de Regressão Logística e limiar calibrado
│
├── notebooks/
│   ├── 01_eda_e_insights.ipynb     # Notebook com Análise Exploratória (Fase 1)
│   └── 02_dashboards_fase4.ipynb   # Notebook com os painéis analíticos em Plotly (Fase 4)
│
├── reports/
│   ├── metrics_report.md   # Relatório formatado em Markdown com métricas finais
│   ├── metrics.json        # Dicionário de métricas estruturadas
│   ├── metrics_summary.csv # Resumo tabular das métricas nas partições
│   ├── roc_curve_test.csv  # Pontos da curva ROC no conjunto de teste
│   └── dashboard_fase4.html # Painel consolidado (Fase 4), abre em qualquer navegador
│
├── src/
│   ├── pipeline.py         # Script de Engenharia de Dados (ETL e Binning)
│   └── train.py            # Script de Treinamento, SMOTE (só no treino), Otimização de Limiar e Avaliação
│
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação principal
```

---

## Instruções de Execução passo a passo

Siga os passos abaixo para configurar o ambiente, executar todo o pipeline do zero e abrir a aplicação web em seu computador.

### 1. Configurar o Ambiente Virtual

Abra o terminal na pasta raiz do projeto e crie/ative o ambiente virtual (`.venv`):

```powershell
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente virtual no PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
# (Ou se preferir executar diretamente o Python do venv: .\.venv\Scripts\python.exe)
```

### 2. Instalar as Dependências

Com o ambiente ativado, instale todas as bibliotecas necessárias:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Executar o Pipeline de Engenharia de Dados (ETL — Fase 2)

O script de ETL lê a base bruta (`data/raw/`), aplica imputação em nulos, cria faixas clínicas para idade (`age_group`) e glicose (`glucose_group`), aplica `StandardScaler` e `OneHotEncoder`, e separa treino/teste (80/20). **Não aplica SMOTE nesta etapa** — `X_train.csv` mantém a distribuição real das classes (~4.8% de AVC) de propósito, para que a validação feita na Fase 3 não seja contaminada por amostras sintéticas:

```powershell
python src/pipeline.py
```
> *Ao finalizar, os arquivos processados (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` e `preprocessor.pkl`) estarão salvos na pasta `data/processed/`, com a distribuição de classes original.*

---

### 4. Treinar o Modelo de Machine Learning (Fase 3)

O pre-processamento gera os conjuntos em `data/processed`. O treino usa diretamente `X_train.csv`, `y_train.csv`, `X_test.csv` e `y_test.csv` dessa pasta; os dados em `data/raw` são usados apenas pelo pré-processamento. A validação é separada de forma estratificada a partir do conjunto de treino (`train_test_split`), **antes** de qualquer balanceamento — assim ela preserva a mesma distribuição de classes do teste.

O **SMOTE** é aplicado nesta etapa (`src/train.py`), exclusivamente sobre a porção efetiva de treino (após o split de validação), para não vazar amostras sintéticas para a validação. O modelo final utiliza `LogisticRegression(class_weight="balanced")` para lidar com o desbalanceamento residual. O limiar de corte (*threshold*) é ajustado dinamicamente na partição de validação (com distribuição real) maximizando o `F1-Score`. A avaliação final no teste utiliza Precision, Recall, F1-Score, ROC AUC, Accuracy e Matriz de Confusão:

```powershell
python src/train.py
```
> *Ao finalizar, o modelo estará exportado em `models/stroke_risk_model.joblib` e os relatórios de métricas e curva ROC estarão atualizados na pasta `reports/`.*

---

### 5. Gerar os Dashboards Analíticos (Fase 4)

O notebook `notebooks/02_dashboards_fase4.ipynb` lê diretamente `data/raw/`, `models/stroke_risk_model.joblib` e os relatórios em `reports/` para construir os painéis de perfil da base, taxas de AVC por variável clínica, correlações, curva ROC, matriz de confusão e importância de variáveis, tudo com **Plotly**. Ao ser executado, ele também exporta um painel consolidado autônomo em `reports/dashboard_fase4.html` (abre em qualquer navegador, sem depender de Jupyter ou Streamlit).

> *Requer `jupyter`/`nbformat`/`ipykernel`/`kaleido` instalados no ambiente para reexecutar o notebook (não incluídos em `requirements.txt` por serem apenas ferramentas de desenvolvimento, não dependências da aplicação). Os gráficos são salvos como imagens estáticas (PNG) no próprio notebook — leve e compatível com qualquer editor — enquanto a versão interativa completa fica em `reports/dashboard_fase4.html`.*

---

### 6. Iniciar a Aplicação Web Streamlit (Fase 5)

Para abrir a interface interativa onde os usuários podem realizar predições ao vivo, visualizar o velocímetro de risco quantitativo e ler a metodologia do projeto:

```powershell
streamlit run app/app.py
```
> *O terminal exibirá uma URL local (geralmente `http://localhost:8501`). O navegador será aberto automaticamente com o sistema quantitativo de suporte à decisão.*

---

## Resumo das Fases do Projeto

1. **Fase 1 (EDA):** Identificação de desbalanceamento severo (`~4.8%` de AVC), nulos no IMC (`bmi`) e análise univariada/bivariada das condições clínicas.
2. **Fase 2 (ETL):** Construção de pipeline automatizado (`build_preprocessor`) com categorização médica em faixas (`pd.cut`), mantendo a distribuição real das classes (sem balanceamento nesta etapa).
3. **Fase 3 (Treinamento):** Treinamento via Scikit-Learn com `SMOTE` aplicado apenas na porção de treino (após separar a validação, para não vazar amostras sintéticas), e limiar (threshold **0.7278**) ajustado na validação com distribuição real, maximizando F1-Score. Métricas no teste: Recall **60.0%** / Precision **20.55%** / F1-Score **0.3061** / ROC AUC **0.8409**.
4. **Fase 4 (Dashboards):** Notebook `02_dashboards_fase4.ipynb` com painéis em `Plotly` sobre perfil da base, outliers, taxas de AVC por variável clínica, correlações, curva ROC, matriz de confusão e importância de variáveis, além do painel consolidado exportado em `reports/dashboard_fase4.html`.
5. **Fase 5 (Web App):** Encapsulamento do modelo em uma aplicação moderna com design adaptativo (Light/Dark Mode) focado em sobriedade técnica e terminologia de engenharia de software.
