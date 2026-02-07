import pandas as pd
import os

# ======= CHANGE ONLY THIS =======
cellType = "cd4t-treg-dnt"  # specify cell type
N = 20   # number of top genes you want to keep
# =================================

# Base folder on D: SSD
base_dir = r"D:\polygence\weight rankings"

# Construct full input path automatically
input_path = os.path.join(
    base_dir,
    cellType,
    f"gene_importance_{cellType}.csv"
)

# Load CSV
df = pd.read_csv(input_path)

# Keep top N
df_topN = df.iloc[:N].copy()

# Construct output path automatically
output_path = os.path.join(
    base_dir,
    cellType,
    f"gene_importance_{cellType}_top{N}.csv"
)

# Save
df_topN.to_csv(output_path, index=False)

print(f"Created new CSV with top {N} genes:\n{output_path}")
