import anndata as ad
import numpy as np
import tensorflow as tf
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
import os
from tqdm import tqdm

# suppress INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        

# ============================== #
# SSD root path and cell type
# ============================== #
ssd_root = r"D:\polygence"
cellType = "nk-ilc"

data_path = os.path.join(ssd_root, "data", cellType + ".h5ad")
output_dir = os.path.join(ssd_root, "weight rankings", cellType)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"gene_importance_{cellType}.csv")

# ============================== #
# read data, setup matrices and classification targets
# ============================== #
adata = ad.read_h5ad(data_path)
if 'log1p' in adata.uns:
    del adata.uns['log1p']
# use top n highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=1800, flavor="seurat")
adata = adata[:, adata.var['highly_variable']]

# Gene expression matrix
X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
y = adata.obs['subject.ageGroup']

# ============================== #
# preprocess data and split train/test
# ============================== #
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ============================== #
# build model and compile
# ============================== #
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================== #
# early stopping
# ============================== #
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ============================== #
# train model 
# ============================== #
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# evaluate model
# ==============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# ==============================
# wrap model for sklearn
# ==============================
class KerasWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return np.argmax(preds, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

wrapped_model = KerasWrapper(model)

# ==============================
# vectorized permutation importance
# ==============================
print("Computing permutation importance with vectorization...")

n_genes = X_test.shape[1]
n_repeats = 4
baseline_acc = wrapped_model.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_acc:.4f}")

importances = np.zeros(n_genes)

for gene_idx in tqdm(range(n_genes), desc="Permuting genes"):
    acc_drops = []
    for _ in range(n_repeats):
        X_shuffled = X_test.copy()
        X_shuffled[:, gene_idx] = np.random.permutation(X_shuffled[:, gene_idx])
        preds = wrapped_model.predict(X_shuffled)
        acc_drops.append(baseline_acc - np.mean(preds == y_test))
    importances[gene_idx] = np.mean(acc_drops)

# Rank genes
indices = np.argsort(importances)[::-1]
gene_names = adata.var_names

print("\nTop 50 most important genes:")
for i in indices[:50]:
    print(f"{gene_names[i]}: {importances[i]:.5f}")

# ==============================
# Save all gene rankings
# ==============================
df = pd.DataFrame({
    "gene": gene_names[indices],
    "importance": importances[indices]
})

df.to_csv(output_file, index=False)
print(f"\nSaved full importance ranking to:\n{output_file}")
