import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def create_binned_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cria faixas clínicas para idade ('age') e glicose ('avg_glucose_level').
    Mantém as colunas contínuas originais e adiciona 'age_group' e 'glucose_group'.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    else:
        X = X.copy()

    if 'age' in X.columns:
        X['age_group'] = pd.cut(
            X['age'],
            bins=[-np.inf, 18, 35, 60, np.inf],
            labels=['Jovem_0-18', 'Adulto_Jovem_19-35', 'Meia_Idade_36-60', 'Idoso_60+']
        ).astype(str)

    if 'avg_glucose_level' in X.columns:
        X['glucose_group'] = pd.cut(
            X['avg_glucose_level'],
            bins=[-np.inf, 100, 125, np.inf],
            labels=['Normal_<100', 'Pre_Diabetes_100-125', 'Elevada_>=126']
        ).astype(str)

    return X

create_binned_features.__module__ = 'pipeline'

def build_preprocessor():
    num_cols_with_na = ['bmi']
    num_cols_no_na = ['age', 'avg_glucose_level']
    binary_cols = ['hypertension', 'heart_disease']
    cat_cols = [
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
        'age_group',
        'glucose_group'
    ]

    num_imputer_scaler = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    num_scaler = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num_na', num_imputer_scaler, num_cols_with_na),
            ('num', num_scaler, num_cols_no_na),
            ('binary', 'passthrough', binary_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    
    preprocessor = Pipeline(steps=[
        ('feature_creation', FunctionTransformer(create_binned_features)),
        ('column_transformer', col_transformer)
    ])
    
    return preprocessor

def run_etl(input_path, output_dir):
    print("Iniciando o processo de ETL...")
    
    df = load_data(input_path)
    
    X = df.drop(columns=['stroke', 'id'], errors='ignore') 
    y = df['stroke']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()

    # SMOTE não é aplicado aqui: X_train.csv mantém a distribuição real (~4.8% AVC) para que
    # o split treino/validação feito em src/train.py não seja contaminado por amostras sintéticas.
    # O balanceamento é aplicado em src/train.py somente na porção efetiva de treino.
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)

    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    print(f"ETL concluído com sucesso! Dados salvos na pasta: {output_dir}")
    print(f"  Tamanho do treino: {X_train_processed.shape[0]} amostras (distribuição original, sem SMOTE)")

if __name__ == "__main__":
    sys.modules["pipeline"] = sys.modules["__main__"]
    INPUT_FILE = "data/raw/healthcare-dataset-stroke-data.csv"
    OUTPUT_DIR = "data/processed/"
    
    run_etl(INPUT_FILE, OUTPUT_DIR)
