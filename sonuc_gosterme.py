# =============================================================================
# egitim_script.py
# =============================================================================
import pandas as pd
import numpy as np
import glob
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb
import lightgbm as lgb

print("Nihai Eğitim Scripti Başlatıldı...")
# =============================================================================
# Adım 2: Veri Yükleme ve Birleştirme
# =============================================================================
try:
    path = r'C:\Users\ataka\OneDrive\Desktop\veri_seti'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_list = [pd.read_csv(f, encoding='latin1', skipinitialspace=True) for f in all_files]
    df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"\nCSV dosyaları birleştirildi.")
except Exception as e:
    print(f"Hata: Veri yüklenirken bir sorun oluştu. {e}")
    exit()

# =============================================================================
# Adım 3: Veri Temizleme ve Ön İşleme
# =============================================================================
df.columns = df.columns.str.strip()
cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Timestamp']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Label'], inplace=True)
label_counts = df['Label'].value_counts()
min_samples_threshold = 20
classes_to_keep = label_counts[label_counts >= min_samples_threshold].index
df = df[df['Label'].isin(classes_to_keep)]
target_encoder = LabelEncoder()
df['Label'] = target_encoder.fit_transform(df['Label'])
y = df['Label']
X = df.drop('Label', axis=1)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Veri ön işleme adımları tamamlandı.")

# =============================================================================
# Adım 4: Model Eğitimi ve Detaylı Metriklerin Toplanması
# =============================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1),
}

detailed_results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\n{'=' * 30}\n{model_name} modeli eğitiliyor...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()


    report_dict = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True,
                                        zero_division=0)


    weighted_avg = report_dict['weighted avg']


    detailed_results.append({
        "Model": model_name,
        "Accuracy": report_dict['accuracy'],
        "Precision": weighted_avg['precision'],
        "Recall": weighted_avg['recall'],
        "F1-Score": weighted_avg['f1-score']
    })

    trained_models[model_name] = model
    print(f"{model_name} modeli {end_time - start_time:.2f} saniyede eğitildi. Doğruluk: {report_dict['accuracy']:.4f}")

# =============================================================================
# Adım 5: Detaylı Sonuçları ve Gerekli Dosyaları Kaydetme
# =============================================================================

results_df = pd.DataFrame(detailed_results).sort_values(by='Accuracy', ascending=False)
results_df.to_csv('model_sonuclari_detayli.csv', index=False)
print("\nDetaylı model sonuçları 'model_sonuclari_detayli.csv' dosyasına kaydedildi.")

joblib.dump(trained_models, 'tum_modeller.pkl')
# ... (diğer dosyaların kaydedilmesi aynı kalabilir) ...

print("\nEğitim Scripti Tamamlandı.")