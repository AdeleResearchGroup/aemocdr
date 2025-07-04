import pandas as pd

# Charger le fichier expression brute (gzip)
df_exp = pd.read_parquet("data/TCGA_data/cell_exp_raw.gzip")

# Extraire les identifiants
patient_ids = df_exp.index.tolist()

print(f" Nombre total de patients : {len(patient_ids)}")

# Sauvegarder la liste dans un fichier texte (un ID par ligne)
with open("TCGA_cell_exp_sample_ids.txt", "w") as f:
    for pid in patient_ids:
        f.write(f"{pid}\n")

print("Saved file : TCGA_cell_exp_sample_ids.txt")
