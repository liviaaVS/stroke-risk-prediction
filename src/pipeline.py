import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def build_preprocessor():
    num_cols_with_na = ['bmi']
    num_cols_no_na = ['age', 'avg_glucose_level']
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_na', num_imputer_scaler, num_cols_with_na),
            ('num', num_scaler, num_cols_no_na),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    
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
    
    print("Aplicando SMOTE para balancear a classe alvo no treino...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train_resampled).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train_resampled.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    
    pd.DataFrame(X_test_processed).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    
    print(f"ETL concluído com sucesso! Dados salvos na pasta: {output_dir}")
    print(f"  Tamanho do treino antes do SMOTE: {X_train_processed.shape[0]} amostras (Desbalanceado)")
    print(f"  Tamanho do treino após o SMOTE: {X_train_resampled.shape[0]} amostras (Balanceado)")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/healthcare-dataset-stroke-data.csv"
    OUTPUT_DIR = "data/processed/"
    
    run_etl(INPUT_FILE, OUTPUT_DIR)
