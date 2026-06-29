import pandas as pd
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Memulai proses re-training model di CI/CD...")
    
    data_path = "dataset_preprocessing/cleaned_data.csv"
    df = pd.read_csv(data_path)
    
    # Membuang kolom string
    X = df.drop(columns=['JobRole', 'JobRole_label', 'Skills', 'YearsExperience'])
    y = df['JobRole_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model Baru: {acc:.4f}")

    output_dir = "model_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    mlflow.sklearn.save_model(model, output_dir)
    print(f"Model berhasil disimpan di direktori {output_dir}/ siap untuk Docker.")

if __name__ == "__main__":
    main()