import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn.functional import normalize
from pretrain_autoencoders import SimpleAutoencoder
from data_loader_pretraining import RawDataLoader
from utils import RAW_BOTH_DATA_FOLDER, DATA_MODALITIES, CCLE_RAW_DATA_FOLDER, TCGA_DATA_FOLDER

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration
CELL_LATENT_DIM = 700
DRUG_LATENT_DIM = 50
ENCODER_CELL_PATH = "encoder_cell.pth"
ENCODER_DRUG_PATH = "encoder_drug.pth"
N_SAMPLES = 200

# Chargement des données brutes (cell + drug)
data_dict, _ = RawDataLoader.load_data(
    data_modalities=DATA_MODALITIES,
    raw_file_directory=RAW_BOTH_DATA_FOLDER,
    screen_file_directory=None,
    sep="\t"
)
X_cell_full, X_drug_full, cell_sizes, drug_sizes = RawDataLoader.get_unique_entities(data_dict)

# Chargement des modèles d'autoencodeurs
encoder_cell = SimpleAutoencoder(X_cell_full.shape[1], CELL_LATENT_DIM).to(device)
encoder_cell.load_state_dict(torch.load(ENCODER_CELL_PATH, map_location=device))
encoder_cell.eval()

encoder_drug = SimpleAutoencoder(X_drug_full.shape[1], DRUG_LATENT_DIM).to(device)
encoder_drug.load_state_dict(torch.load(ENCODER_DRUG_PATH, map_location=device))
encoder_drug.eval()

# Conversion et normalisation des tenseurs
X_cell_tensor = normalize(torch.tensor(X_cell_full.values, dtype=torch.float32), dim=0).to(device)
X_drug_tensor = normalize(torch.tensor(X_drug_full.values, dtype=torch.float32), dim=0).to(device)

# Sous-échantillonnage indépendant pour cell et drug
cell_indices = torch.randperm(X_cell_tensor.shape[0])[:min(N_SAMPLES, X_cell_tensor.shape[0])]
drug_indices = torch.randperm(X_drug_tensor.shape[0])[:min(N_SAMPLES, X_drug_tensor.shape[0])]

X_cell_tensor_sample = X_cell_tensor[cell_indices]
X_drug_tensor_sample = X_drug_tensor[drug_indices]

# Encodage latent
with torch.no_grad():
    Z_cell = encoder_cell.encoder(X_cell_tensor).cpu().numpy()
    Z_drug = encoder_drug.encoder(X_drug_tensor).cpu().numpy()

print(Z_cell.shape)
print(Z_drug.shape)

# t-SNE sur les données brutes
tsne_raw_cell = TSNE(n_components=2, random_state=42).fit_transform(X_cell_tensor.cpu().numpy())
tsne_raw_drug = TSNE(n_components=2, random_state=42).fit_transform(X_drug_tensor.cpu().numpy())

# t-SNE sur les données latentes
tsne_latent_cell = TSNE(n_components=2, random_state=42).fit_transform(Z_cell)

tsne_latent_drug = TSNE(n_components=2, random_state=42).fit_transform(Z_drug)

# Affichage
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].scatter(tsne_raw_cell[:, 0], tsne_raw_cell[:, 1], s=5)
axs[0, 0].set_title("t-SNE - Cell Raw")
axs[0, 1].scatter(tsne_raw_drug[:, 0], tsne_raw_drug[:, 1], s=5)
axs[0, 1].set_title("t-SNE - Drug Raw")
axs[1, 0].scatter(tsne_latent_cell[:, 0], tsne_latent_cell[:, 1], s=5)
axs[1, 0].set_title("t-SNE - Cell Latent")
axs[1, 1].scatter(tsne_latent_drug[:, 0], tsne_latent_drug[:, 1], s=5)
axs[1, 1].set_title("t-SNE - Drug Latent")
plt.tight_layout()
plt.show()