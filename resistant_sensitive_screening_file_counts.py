import pandas as pd

# Charger le fichier .tsv
file_path = "CTRP_GDSC_data/CTRP_GDSC_data/drug_screening_matrix_gdsc_ctrp.tsv"
df = pd.read_csv(file_path, sep='\t', index_col=0)

# Compter les occurrences de chaque valeur dans toute la matrice
value_counts = df.stack().value_counts()  # .stack() pour transformer en Series à 1D

# Print results
print("CTRP-GDSC dataset: screening file (drug_screening_matrix_gdsc_ctrp.tsv)")
print("Nombre de -1 (sensitive) :", value_counts.get(-1, 0))
print("Nombre de 0 (not tested / unknown) :", value_counts.get(0, 0))
print("Nombre de 1 (resistant) :", value_counts.get(1, 0))

# Charger le fichier .tsv
file_path = "CCLE_data/CCLE_data/drug_screening_matrix_ccle.tsv"
df = pd.read_csv(file_path, sep='\t', index_col=0)

# Compter les occurrences de chaque valeur dans toute la matrice
value_counts = df.stack().value_counts()  # .stack() pour transformer en Series à 1D

# Print results
print("CCLE dataset: screening file (drug_screening_matrix_ccle.tsv)")
print("Nombre de -1 (sensitive) :", value_counts.get(-1, 0))
print("Nombre de 0 (not tested / unknown) :", value_counts.get(0, 0))
print("Nombre de 1 (resistant) :", value_counts.get(1, 0))