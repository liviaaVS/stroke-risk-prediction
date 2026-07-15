# stroke-risk-prediction

Projeto de classificacao para predicao de risco de AVC usando Scikit-Learn.

## Como executar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\pipeline.py
.\.venv\Scripts\python.exe src\train.py
```

## Entregaveis gerados

- Modelo final: `models/stroke_risk_model.joblib`
- Relatorio principal: `reports/metrics_report.md`
- Metricas estruturadas: `reports/metrics.json` e `reports/metrics_summary.csv`
- Curva ROC do teste: `reports/roc_curve_test.csv`

O pre-processamento gera os conjuntos em `data/processed`. O treino usa
diretamente `X_train.csv`, `y_train.csv`, `X_test.csv` e `y_test.csv` dessa
pasta; os dados em `data/raw` sao usados apenas pelo pre-processamento. A
validacao e separada de forma estratificada a partir do conjunto de treino.
O modelo final usa regressao logistica com `class_weight="balanced"` para lidar
com o desbalanceamento da classe alvo. O limiar de classificacao e ajustado na
validacao maximizando `F1-Score`. A avaliacao final usa Precision, Recall,
F1-Score, ROC AUC, Accuracy e matriz de confusao.
