import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from autoencoder import Autoencoder, trainAutoencoder
from data_loader import RawDataLoader
from utils import RAW_BOTH_DATA_FOLDER, BOTH_SCREENING_DATA_FOLDER, TCGA_DATA_FOLDER, TCGA_SCREENING_DATA, DRUG_DATA_FOLDER

# Charger les données cellulaires via RawDataLoader.load_data()

data_modalities = ['cell_exp', 'drug_finger']
data_dict, screening = RawDataLoader.load_data(
    data_modalities=data_modalities,
    raw_file_directory=RAW_BOTH_DATA_FOLDER,
    screen_file_directory=BOTH_SCREENING_DATA_FOLDER,  
    sep="\t",
    drug_directory=DRUG_DATA_FOLDER
)

print(screening.shape)
print(screening.head())

# Préparer X_cell
def prepare_cell_input_data(data_dict, screening):

    # Trouver les indices où screening == 1 (résistance) et -1 (sensibilité)
    resistance = np.argwhere(screening.to_numpy() == 1).tolist()
    resistance.sort(key=lambda x: (x[0], x[1]))  # trier par ligne puis colonne
    resistance = np.array(resistance)
    
    sensitive = np.argwhere(screening.to_numpy() == -1).tolist()
    sensitive.sort(key=lambda x: (x[0], x[1]))
    sensitive = np.array(sensitive)
    
    print(f"sensitive data len: {len(sensitive)}")
    print(f"resistance data len: {len(resistance)}")
    
    # Construire le DataFrame pour les données cellulaires en ne gardant que les modalités qui commencent par 'cell'
    cell_data_types = list(filter(lambda x: x.startswith('cell'), data_dict.keys()))
    cell_data_types.sort()
    cell_data = pd.concat(
        [pd.DataFrame(data_dict[data_type].add_suffix(f'_{data_type}'), dtype=np.float32)
         for data_type in cell_data_types],
        axis=1
    )
    print(f"Taille de cell_data: {cell_data.shape}")
    
    # Extraire les échantillons de résistance et sensibilité
    Xp_cell = cell_data.iloc[resistance[:, 0], :].reset_index(drop=True)
    Xp_cell.index = [f'{screening.index[x[0]]}' for x in resistance]
    
    Xn_cell = cell_data.iloc[sensitive[:, 0], :].reset_index(drop=True)
    Xn_cell.index = [f'{screening.index[x[0]]}' for x in sensitive]
    
    # Concaténer les échantillons pour avoir l'ensemble complet
    X_cell = pd.concat([Xp_cell, Xn_cell])
    
    # Créer le vecteur de labels : 0 pour résistance, 1 pour sensibilité
    Y = np.append(np.zeros(resistance.shape[0]), np.ones(sensitive.shape[0]))
    
    return X_cell, Y

X_cell, Y = prepare_cell_input_data(data_dict, screening)
print(f"Taille de X_cell: {X_cell.shape}")

# Préparer X_cell pour t-SNE
# Transformer X_cell en tenseur PyTorch et normaliser
cell_tensor = torch.tensor(X_cell.values, dtype=torch.float32)
cell_tensor = torch.nn.functional.normalize(cell_tensor, dim=0)

# Entraîner l'autoencodeur sur l'ensemble des données cellulaires
latent_dim = 50  
input_dim = X_cell.shape[1]
autoencoder_cell = Autoencoder(input_dim, latent_dim)

# Créer un TensorDataset et un DataLoader
dataset_cell = TensorDataset(cell_tensor)
batch_size = 64
data_loader_cell = DataLoader(dataset_cell, batch_size=batch_size, shuffle=True)

# Définir le nombre d'epoch d'entraînement
num_epochs = 25

# Entraîner l'autoencodeur sur l'ensemble complet des données
trainAutoencoder(autoencoder_cell, data_loader_cell, data_loader_cell, num_epochs, name='_cell')

# Extraire la représentation latente

# Passer l'autoencodeur en mode évaluation
autoencoder_cell.eval()

# Calculer la représentation latente en passant l'ensemble complet des données par l'encodeur
with torch.no_grad():
    latent_representation_cell = autoencoder_cell.encoder(cell_tensor)

# Convertir la représentation latente en tableau NumPy pour l'utiliser avec t-SNE
latent_representation_cell = latent_representation_cell.numpy()

# Appliquer t-SNE sur les données d'entrée normalisées (avant encodage)
tsne_input_cell = TSNE(n_components=2, random_state=42)
X_tsne_input_cell = tsne_input_cell.fit_transform(X_cell)

# Appliquer t-SNE sur les représentations latentes (après encodage)
tsne_latent_cell = TSNE(n_components=2, random_state=42)
X_tsne_latent_cell = tsne_latent_cell.fit_transform(latent_representation_cell)

# Visualiser les résultats

plt.figure(figsize=(12, 6))

# Ajouter un titre
plt.suptitle("CTRP_GDSC - ['cell_exp', 'drug_finger']", fontsize=16)

# Sous-figure pour la projection t-SNE sur les données d'entrée
plt.subplot(1, 2, 1)
for class_value, color, label in zip([0, 1], ['blue', 'red'], ['Résistance', 'Sensibilité']):
    mask = (Y == class_value)
    plt.scatter(
        X_tsne_input_cell[mask, 0],
        X_tsne_input_cell[mask, 1],
        c=color,
        s=10,
        label=label
    )
plt.title("t-SNE cell (input data)")
plt.legend()

# Sous-figure pour la projection t-SNE sur l'espace latent
plt.subplot(1, 2, 2)
for class_value, color, label in zip([0, 1], ['blue', 'red'], ['Résistance', 'Sensibilité']):
    mask = (Y == class_value)
    plt.scatter(
        X_tsne_latent_cell[mask, 0],
        X_tsne_latent_cell[mask, 1],
        c=color,
        s=10,
        label=label
    )
plt.title("t-SNE cell (latent space)")
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()