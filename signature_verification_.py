import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Configuration
LABEL_MAP = {'Genuine': 1, 'Forged': 0}
ALL_FEATURES = ['X', 'Y', 'P', 'al', 'az']

# Dataset pairs
DATASETS = [
    ('MCYT', 'mcytTraining.csv', 'mcytTesting.csv'),
    ('SVC', 'svcTraining.csv', 'svcTesting.csv'),
    ('Chinese', 'chineseTraining.csv', 'chineseTesting.csv'),
    ('Dutch', 'dutchTraining.csv', 'dutchTesting.csv'),
    ('German', 'germanTraining.csv', 'germanTesting.csv')
]

# Models
MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC(kernel='rbf', probability=True)
}

results = []

# Loop through datasets
for dataset_name, train_file, test_file in DATASETS:
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()
    df_train['signatureOrigin'] = df_train['signatureOrigin'].str.strip()
    df_test['signatureOrigin'] = df_test['signatureOrigin'].str.strip()

    df_train['Label'] = df_train['signatureOrigin'].map(LABEL_MAP)
    df_test['Label'] = df_test['signatureOrigin'].map(LABEL_MAP)

    if dataset_name == 'MCYT':
        group_cols = ['SigID', 'signatureOrigin', 'Label']
        df_train = df_train.groupby(group_cols, as_index=False)[ALL_FEATURES].mean()
        df_test = df_test.groupby(group_cols, as_index=False)[ALL_FEATURES].mean()

    valid_features = [f for f in ALL_FEATURES if df_train[f].notna().sum() > 0 and df_test[f].notna().sum() > 0]
    df_train = df_train.dropna(subset=valid_features)
    df_test = df_test.dropna(subset=valid_features)

    X_train = df_train[valid_features].values
    y_train = df_train['Label'].values
    X_test = df_test[valid_features].values
    y_test = df_test['Label'].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 50)
    print(f"Results for Dataset: {dataset_name}")
    print("=" * 50)

    for model_name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) else 0
        frr = fn / (fn + tp) if (fn + tp) else 0
        eer = (far + frr) / 2

        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Features': valid_features,
            'Accuracy': acc,
            'Precision': precision,
            'F1': f1,
            'FAR': far,
            'FRR': frr,
            'EER': eer,
            'ConfusionMatrix': cm
        })

        print(f"Dataset: {dataset_name}")
        print(f"Features used: {', '.join(valid_features)}")
        print(f"Model: {model_name}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"FAR: {far:.4f}")
        print(f"FRR: {frr:.4f}")
        print(f"EER: {eer:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # Confusion Matrix Heatmap
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        print("--------------------------------------------------")

    # PCA visualization (once per dataset)
    X_combined = np.vstack((X_train_scaled, X_test_scaled))
    y_combined = np.concatenate((y_train, y_test))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_combined, cmap='bwr', alpha=0.6)
    plt.title(f'PCA (Train+Test) - {dataset_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
