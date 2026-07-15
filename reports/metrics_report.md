# Relatorio de desempenho - Predicao de AVC

## Separacao dos dados

| Particao | Classe 0 | Classe 1 | Total |
| --- | ---: | ---: | ---: |
| treino | 2916 | 2917 | 5833 |
| validacao | 973 | 972 | 1945 |
| teste | 972 | 50 | 1022 |

## Validacao

Modelo: `logistic_regression_balanced`

| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3570 | 0.7337 | 0.9352 | 0.8223 | 0.8462 | 0.7979 |

## Teste final

| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3570 | 0.1123 | 0.8400 | 0.1981 | 0.8451 | 0.6673 |

## Matriz de confusao no teste

|  | Previsto 0 | Previsto 1 |
| --- | ---: | ---: |
| Real 0 | 640 | 332 |
| Real 1 | 8 | 42 |

## Artefatos

- Modelo final: `models/stroke_risk_model.joblib`
- Curva ROC do teste: `reports/roc_curve_test.csv`
