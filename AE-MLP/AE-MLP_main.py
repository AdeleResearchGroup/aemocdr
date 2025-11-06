# =============================
# AE-MLP (DeepDRA derived) — Main script
# =============================

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from DeepDRA import DeepDRA, train, test
from data_loader import RawDataLoader
from evaluation import Evaluation
from utils import *
import random
import torch
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the batch size for training
batch_size = 64

# Instantiate the combined model
num_epochs = 25

cell_latent_dim = 700
drug_latent_dim = 50

def tsne_unlabeled(X_np, title):
    np.random.seed(42)
    emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_np)
    plt.figure(figsize=(7,5))
    plt.scatter(emb[:,0], emb[:,1], s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel("TSNE-1"); plt.ylabel("TSNE-2")
    plt.grid(True); plt.tight_layout(); plt.show()

def knn_radius_stats(Z: np.ndarray, k: int = 10):
    """
    In the latent space Z (n_samples, d), compute the local radius defined as
    the distance to the k-th nearest neighbor, then return:
    - mean_radius (↓ = better),
    - std_radius,
    - cv_radius = std/mean (↓ = more homogeneous structure).

    We standardize Z (per feature) to enable fair comparison across models.
    """
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    # Standardize latent features
    Zs = StandardScaler().fit_transform(Z)

    # Use k+1 neighbors because the 0-th neighbor is the point itself (distance 0)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nbrs.fit(Zs)
    dists, _ = nbrs.kneighbors(Zs)

    radii = dists[:, k]  # distance to the k-th neighbor
    mean_r = float(np.mean(radii))
    std_r  = float(np.std(radii))
    cv_r   = float(std_r / (mean_r + 1e-12))
    return mean_r, std_r, cv_r

@torch.no_grad()
def visualize_unique_latents_from_labeled(model, x_cell_train_df, x_drug_train_df,
                                          cell_norms, drug_norms, device):
    """
    t-SNE visualization of unique entities (cells/drugs) but
    ONLY from labeled data (training pairs).
    Encode via the model => "encoded cells/drugs".
    """

    # Unique entities from the LABELED PAIRS
    cell_unique_df = x_cell_train_df.drop_duplicates()
    drug_unique_df = x_drug_train_df.drop_duplicates()

    # Tensors + same normalization as training (no leakage)
    x_cell_u = torch.tensor(cell_unique_df.values, dtype=torch.float32)
    x_drug_u = torch.tensor(drug_unique_df.values, dtype=torch.float32)

    # apply EXACTLY the same norms learned on training data
    x_cell_u = (x_cell_u / cell_norms).to(device)
    x_drug_u = (x_drug_u / drug_norms).to(device)

    model.eval()
    B = 512

    # Encoding UNIQUE CELLS (encoded cells)
    zc_list = []
    for i in range(0, x_cell_u.size(0), B):
        xb_cell = x_cell_u[i:i+B]
        # dummy drug with the correct shape (same #features as training)
        dummy_drug = torch.zeros((xb_cell.size(0), x_drug_u.shape[1]), device=device)
        _, _, _, (zc_b, _) = model(xb_cell, dummy_drug, return_latent=True)
        zc_list.append(zc_b.detach().cpu())
    z_cell_unique = torch.cat(zc_list, dim=0).numpy()

    # Encoding UNIQUE DRUGS (encoded drugs)
    zd_list = []
    for j in range(0, x_drug_u.size(0), B):
        xb_drug = x_drug_u[j:j+B]
        dummy_cell = torch.zeros((xb_drug.size(0), x_cell_u.shape[1]), device=device)
        _, _, _, (_, zd_b) = model(dummy_cell, xb_drug, return_latent=True)
        zd_list.append(zd_b.detach().cpu())
    z_drug_unique = torch.cat(zd_list, dim=0).numpy()

    # t-SNE on "encoded" latents, derived only from labeled training pairs
    tsne_unlabeled(z_cell_unique, "t-SNE - encoded cells (train data)")
    tsne_unlabeled(z_drug_unique, "t-SNE - encoded drugs (train data)")

    print("z_cell shape:", z_cell_unique.shape)
    print("z_drug shape:", z_drug_unique.shape)
    # unsupervised metric: k-NN radius (k=10) on latents
    c_mean, c_std, c_cv = knn_radius_stats(z_cell_unique, k=10)
    d_mean, d_std, d_cv = knn_radius_stats(z_drug_unique, k=10)
    print(f"[kNN radius | Cells] mean={c_mean:.4f}, std={c_std:.4f}, cv={c_cv:.4f}")
    print(f"[kNN radius | Drugs] mean={d_mean:.4f}, std={d_std:.4f}, cv={d_cv:.4f}")

def tsne_labeled(z_tensor, y_tensor, title):
    z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.numpy())
    y_np = y_tensor.detach().cpu().numpy().ravel().astype(int)
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(z_embedded[y_np == label, 0], z_embedded[y_np == label, 1],
                    label='Resistant' if label == 0 else 'Sensitive',
                    c=color, s=10, alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device, visualize='first', run_id=0, return_raw=False):
    """
    Train and evaluate the DeepDRA model.

    Parameters:
    - X_cell_train (pd.DataFrame): Training data for the cell modality.
    - X_cell_test (pd.DataFrame): Test data for the cell modality.
    - X_drug_train (pd.DataFrame): Training data for the drug modality.
    - X_drug_test (pd.DataFrame): Test data for the drug modality.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - cell_sizes (list): Sizes of the cell modality features.
    - drug_sizes (list): Sizes of the drug modality features.

    Returns:
    - result: Evaluation result on the test set.
    """

    model = DeepDRA(cell_sizes, drug_sizes, cell_latent_dim, drug_latent_dim).to(device)

    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    n_cell  = x_cell_train.shape[1]
    y_all   = np.asarray(y_train).ravel()

    # Stratified split of indices
    idx = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=RANDOM_SEED, stratify=y_all)

    # Normalize on training data
    thr = 1e-6
    X_train = X_pairs.iloc[train_idx]
    x_cell_train_tensor = torch.tensor(X_train.iloc[:, :n_cell].values, dtype=torch.float32)
    x_drug_train_tensor = torch.tensor(X_train.iloc[:,  n_cell:].values, dtype=torch.float32)
    
    cell_norms = torch.norm(x_cell_train_tensor, dim=0, keepdim=True)
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    
    drug_norms = torch.norm(x_drug_train_tensor, dim=0, keepdim=True)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)

    # save train norms for reproducibility
    torch.save(cell_norms.detach().cpu(), f"train_cell_l2norms_run_{run_id}.pt")
    torch.save(drug_norms.detach().cpu(), f"train_drug_l2norms_run_{run_id}.pt")
    
    # RandomUnderSampler on the training set
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = rus.fit_resample(X_pairs.iloc[train_idx], y_all[train_idx])

    # Reconstruct (cell/drug) for train/val
    x_cell_train = X_train_bal.iloc[:, :n_cell]
    x_drug_train = X_train_bal.iloc[:, n_cell:]
    y_train = y_train_bal

    x_cell_val = X_pairs.iloc[val_idx, :n_cell]
    x_drug_val = X_pairs.iloc[val_idx, n_cell:]
    y_val = y_all[val_idx]

    print(f"x_cell_train shape (after RUS): {x_cell_train.shape}")
    print(f"x_drug_train shape (after RUS): {x_drug_train.shape}")
    print(f"x_cell_val shape:  {x_cell_val.shape}")
    print(f"x_drug_val shape:  {x_drug_val.shape}")
    print(f"y_train counts (after RUS): "
          f"0={np.sum(y_train_bal==0)}, 1={np.sum(y_train_bal==1)}")

    # Convert training data to PyTorch tensors
    x_cell_train_tensor = torch.Tensor(x_cell_train.values) / cell_norms
    x_drug_train_tensor = torch.Tensor(x_drug_train.values) / drug_norms 
    x_cell_val_tensor = torch.Tensor(x_cell_val.values) / cell_norms
    x_drug_val_tensor = torch.Tensor(x_drug_val.values) / drug_norms 

    y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
    y_val_tensor = torch.Tensor(y_val).unsqueeze(1)

    # Send tensors to device
    x_cell_train_tensor = x_cell_train_tensor.to(device); x_drug_train_tensor = x_drug_train_tensor.to(device); y_train_tensor = y_train_tensor.to(device)
    x_cell_val_tensor = x_cell_val_tensor.to(device); x_drug_val_tensor = x_drug_val_tensor.to(device); y_val_tensor = y_val_tensor.to(device)
    
    # Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_cell_val_tensor, x_drug_val_tensor, y_val_tensor)
    
    # Create the train_loader and val_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train(model, train_loader, val_loader, num_epochs, class_weights=None)

    if visualize in ('always',) or (visualize == 'first' and run_id == 0):
        visualize_unique_latents_from_labeled(
            model,
            x_cell_train, x_drug_train,   # DataFrames of TRAIN PAIRS (labeled)
            cell_norms, drug_norms,
            device
        )
        
    model.eval()

    # Convert test data to PyTorch tensors
    x_cell_test_tensor = torch.Tensor(x_cell_test.values).to(device)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values).to(device)
    y_test_tensor = torch.Tensor(y_test).to(device)

    cell_norms = cell_norms.to(device)
    drug_norms = drug_norms.to(device)

    # Normalize test set using training norms
    x_cell_test_tensor = x_cell_test_tensor / cell_norms
    x_drug_test_tensor = x_drug_test_tensor / drug_norms


    # Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test), shuffle=False)


    # t-SNE (never / first / always)
    if visualize == 'always' or (visualize == 'first' and run_id == 0):
        with torch.no_grad():
            model.eval()
            # Encode latents
            enc_dataloader = DataLoader(TensorDataset(x_cell_train_tensor, x_drug_train_tensor), batch_size=512, shuffle=False)
            zc_list, zd_list = [], []
            for xb_c, xb_d in enc_dataloader:
                _, _, _, (zc_b, zd_b) = model(xb_c, xb_d, return_latent=True)
                zc_list.append(zc_b.detach().cpu())
                zd_list.append(zd_b.detach().cpu())
            z_cell_fold = torch.cat(zc_list, dim=0)
            z_drug_fold = torch.cat(zd_list, dim=0)

        # z_cell / z_drug
        tsne_labeled(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
        tsne_labeled(z_drug_fold, y_train_tensor, "t-SNE - z_drug")

        # Concatenated: blue/red by labels
        tsne_labeled(torch.cat([z_cell_fold, z_drug_fold], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    # Extract submodules 
    enc_cell = model.cell_autoencoder.encoder     # cell encoder
    enc_drug = model.drug_autoencoder.encoder     # drug encoder
    mlp_head = model.mlp                          # MLP head
    
    # Save the encoders (modules) + the MLP (state_dict)
    run_id = 0  # adapte si tu fais des runs multiples
    torch.save(enc_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(enc_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(mlp_head.state_dict(), f"mlp_trained_run_{run_id}.pth")

    # Test the model
    return test(model, test_loader, show_plot=True, return_raw=return_raw)

def cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=2, visualize='first', run_id=0):

    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    # Concatenate to split/resample by pair
    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    
    # Labels as 0/1
    y_np = np.asarray(y_train).ravel()
    
    # StratifiedKFold keep 0/1 proportion
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_data, val_data) in enumerate(cv.split(np.zeros(len(y_np)), y_np)):
        print('Fold {}'.format(fold + 1))

        # split by pairs
        X_train, y_train = X_pairs.iloc[train_data], y_np[train_data]
        X_val, y_val = X_pairs.iloc[val_data],  y_np[val_data]
        
        n_cell = x_cell_train.shape[1]

        x_cell_train_tensor = torch.tensor(X_train.iloc[:, :n_cell].values, dtype=torch.float32)
        x_drug_train_tensor = torch.tensor(X_train.iloc[:, n_cell:].values, dtype=torch.float32)
        
        # Normalize on train data
        thr = 1e-6
        cell_norms = torch.norm(x_cell_train_tensor, dim=0, keepdim=True)
        cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
        
        drug_norms = torch.norm(x_drug_train_tensor, dim=0, keepdim=True)
        drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)
    
        # save train norms for reproducibility
        torch.save(cell_norms.detach().cpu(), f"train_cell_l2norms_run_{run_id}.pt")
        torch.save(drug_norms.detach().cpu(), f"train_drug_l2norms_run_{run_id}.pt")
        
        # RUS on train fold
        rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
        X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

        # Separate cells and drugs
        x_cell_train = X_train_bal.iloc[:, :n_cell]
        x_drug_train = X_train_bal.iloc[:, n_cell:]
        x_cell_val = X_val.iloc[:, :n_cell]
        x_drug_val = X_val.iloc[:, n_cell:]

        print(f"[Fold {fold+1}] x_cell_train shape (after RUS): {x_cell_train.shape}")
        print(f"[Fold {fold+1}] x_drug_train shape (after RUS): {x_drug_train.shape}")
        print(f"[Fold {fold+1}] x_cell_val shape:  {x_cell_val.shape}")
        print(f"[Fold {fold+1}] x_drug_val shape:  {x_drug_val.shape}")
        print(f"[Fold {fold+1}] y_train counts (after RUS): "
              f"0={np.sum(y_train_bal==0)}, 1={np.sum(y_train_bal==1)}")

        # Tensors
        x_cell_train_tensor = torch.Tensor(x_cell_train.values) / cell_norms
        x_drug_train_tensor = torch.Tensor(x_drug_train.values) / drug_norms   
        x_cell_val_tensor = torch.Tensor(x_cell_val.values) / cell_norms
        x_drug_val_tensor = torch.Tensor(x_drug_val.values) / drug_norms   
        y_train_tensor = torch.Tensor(y_train_bal).unsqueeze(1)
        y_val_tensor = torch.Tensor(y_val).unsqueeze(1)
        
        # Send to device
        x_cell_train_tensor = x_cell_train_tensor.to(device); x_drug_train_tensor = x_drug_train_tensor.to(device); y_train_tensor = y_train_tensor.to(device)
        x_cell_val_tensor = x_cell_val_tensor.to(device); x_drug_val_tensor = x_drug_val_tensor.to(device); y_val_tensor = y_val_tensor.to(device)
        
        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
        val_dataset   = TensorDataset(x_cell_val_tensor,  x_drug_val_tensor, y_val_tensor)
        
        # Create the train_loader and val_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # Initialize model 
        model = DeepDRA(cell_sizes, drug_sizes, cell_latent_dim, drug_latent_dim).to(device)

        # Display model
        if fold==0:
            print("\nModel architecture:\n")
            print(model)

        # Train model
        train(model, train_loader, val_loader, num_epochs, class_weights=None)

        if visualize in ('always',) or (visualize == 'first' and fold == 0):
            visualize_unique_latents_from_labeled(
                model,
                x_cell_train, x_drug_train,   # DataFrames des PAIRES TRAIN (étiquetées)
                cell_norms, drug_norms,
                device
            )
        # Evaluate on the entire validation fold in a single pass
        val_loader_full = DataLoader(val_dataset, batch_size=len(y_val_tensor), shuffle=False)
        results = test(model, val_loader_full, show_plot=False)
        
        # Add results to the history dictionary
        Evaluation.add_results(history, results)

        # T-SNE (never / first / always)
        if visualize == 'always' or (visualize == 'first' and fold == 0):
            with torch.no_grad():
                model.eval()
                # Encode latent
                enc_dataloader = DataLoader(TensorDataset(x_cell_train_tensor, x_drug_train_tensor), batch_size=512, shuffle=False)
                zc_list, zd_list = [], []
                for xb_c, xb_d in enc_dataloader:
                    _, _, _, (zc_b, zd_b) = model(xb_c, xb_d, return_latent=True)
                    zc_list.append(zc_b.detach().cpu())
                    zd_list.append(zd_b.detach().cpu())
                z_cell_fold = torch.cat(zc_list, dim=0)
                z_drug_fold = torch.cat(zd_list, dim=0)

            # z_cell / z_drug
            tsne_labeled(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
            tsne_labeled(z_drug_fold, y_train_tensor, "t-SNE - z_drug")

            # concatenated: blue/red by labels
            tsne_labeled(torch.cat([z_cell_fold, z_drug_fold], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    
    # Extract submodules 
    enc_cell = model.cell_autoencoder.encoder     # cell encoder
    enc_drug = model.drug_autoencoder.encoder     # drug encoder
    mlp_head = model.mlp                          # MLP head
    
    # Save the encoders (modules) + the MLP (state_dict)
    run_id = 0  # adapte si tu fais des runs multiples
    torch.save(enc_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(enc_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(mlp_head.state_dict(), f"mlp_trained_run_{run_id}.pth")
    
    return history

def run(k, is_test=False ):
    """
    Run the training and evaluation process k times.

    Parameters:
    - k (int): Number of times to run the process.
    - is_test (bool): If True, run on test data; otherwise, perform train-validation split.

    Returns:
    - history (dict): Dictionary containing evaluation metrics for each run.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Initialize a dictionary to store evaluation metrics
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}
    
    # Load training data
    train_data, train_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                            raw_file_directory=RAW_BOTH_DATA_FOLDER,
                                                            screen_file_directory=BOTH_SCREENING_DATA_FOLDER,
                                                            sep="\t")

    print('train_data when loaded:', train_data.keys())
    for key, df in train_data.items():
        print(f"{key}: {df.shape}")

    
    # Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=TCGA_DATA_FOLDER,
                                                              screen_file_directory=TCGA_SCREENING_DATA,
                                                              sep="\t")

        print('test_data when loaded:', test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")
        
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)


        print('train_data after feature intersection with test set:', train_data.keys())
        for key, df in train_data.items():
            print(f"{key}: {df.shape}")

    # Save the feature columns for reproducibility:
    all_features = {}
    for key, df in train_data.items():
        all_features[key] = df.columns.tolist()
        
    import pickle
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(all_features, f)
    
    # Prepare input data for training
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                   train_drug_screen)

    

    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)
    all_runs = []
    
    # Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        if is_test:

            # Train and evaluate the DeepDRA model on test data
            results, y_true, y_score = train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device, visualize='first', run_id=i, return_raw=True)
            # Display final results
            Evaluation.add_results(history, results)
            
            all_runs.append((y_true, y_score))

        else:

            results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5, visualize='first', run_id=i)
            if isinstance(results.get('AUC', None), list):
                for m in history:
                    history[m].extend(results[m])
            else:
                Evaluation.add_results(history, results)

    if is_test and len(all_runs) > 0:
        fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std = Evaluation.aggregate_roc(all_runs, n_points=200)
        rec_grid, mean_prec, std_prec, auprc_mean, auprc_std = Evaluation.aggregate_pr(all_runs, n_points=200)

        Evaluation.plot_mean_roc(fpr_grid, mean_tpr, std_tpr, auc_mean, auc_std, label="AE-MLP")
        Evaluation.plot_mean_pr(rec_grid, mean_prec, std_prec, auprc_mean, auprc_std, label="AE-MLP")


    
    # Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=True)

