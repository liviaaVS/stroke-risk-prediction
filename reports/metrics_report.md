# Relatorio de desempenho - Predicao de AVC

## Separacao dos dados

| Particao | Classe 0 | Classe 1 | Total |
| --- | ---: | ---: | ---: |
| treino | 2917 | 149 | 3066 |
| validacao | 972 | 50 | 1022 |
| teste | 972 | 50 | 1022 |

## Validacao

Modelo: `logistic_regression_balanced`

| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7278 | 0.1800 | 0.5400 | 0.2700 | 0.8267 | 0.8571 |

## Teste final

| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7278 | 0.2055 | 0.6000 | 0.3061 | 0.8409 | 0.8669 |

## Matriz de confusao no teste

|  | Previsto 0 | Previsto 1 |
| --- | ---: | ---: |
| Real 0 | 856 | 116 |
| Real 1 | 20 | 30 |

## Artefatos

- Modelo final: `models/stroke_risk_model.joblib`
- Curva ROC do teste: `reports/roc_curve_test.csv`
