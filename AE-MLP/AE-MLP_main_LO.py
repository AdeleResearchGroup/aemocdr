# =============================
# AE-MLP (DeepDRA derived) — Main LO script
# =============================

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold

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

def parse_cell_ids_from_pairs(index_like):
    s = pd.Index(index_like).to_series()
    # Capture everything before the first comma: "(CELL,DRUG)" -> "CELL"
    return s.str.extract(r'^\(([^,]+),')[0].values

def parse_drug_ids_from_pairs(index_like):
    s = pd.Index(index_like).to_series()
    # Capture everything after the comma: "(CELL,DRUG)" -> "DRUG"
    return s.str.extract(r'^\([^,]+,\s*([^)]+)\)')[0].values

def train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device, split='LDO'):
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
    # Group for LDO/LCO (LDO: by drug ID or LCO: by cell ID)
    if split.upper() == 'LDO':
        groups = parse_drug_ids_from_pairs(x_cell_train.index)
    elif split.upper() == 'LCO':
        groups = parse_cell_ids_from_pairs(x_cell_train.index)

    model = DeepDRA(cell_sizes, drug_sizes, cell_latent_dim, drug_latent_dim).to(device)

    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    n_cell  = x_cell_train.shape[1]
    y_all   = np.asarray(y_train).ravel()

    # Split unique, group-aware et stratifié
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(y_all)), y_all, groups))

    # Normalize on training data
    thr = 1e-6
    X_train = X_pairs.iloc[train_idx]
    x_cell_train_tensor = torch.tensor(X_train.iloc[:, :n_cell].values, dtype=torch.float32)
    x_drug_train_tensor = torch.tensor(X_train.iloc[:,  n_cell:].values, dtype=torch.float32)
    
    cell_norms = torch.norm(x_cell_train_tensor, dim=0, keepdim=True)
    cell_norms = torch.where(cell_norms < thr, torch.ones_like(cell_norms), cell_norms)
    
    drug_norms = torch.norm(x_drug_train_tensor, dim=0, keepdim=True)
    drug_norms = torch.where(drug_norms < thr, torch.ones_like(drug_norms), drug_norms)

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

    model.eval()

    # Convert test data to PyTorch tensors
    x_cell_test_tensor = torch.Tensor(x_cell_test.values).to(device)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values).to(device)
    y_test_tensor = torch.Tensor(y_test).to(device)

    cell_norms = cell_norms.to(device)
    drug_norms = drug_norms.to(device)

    # normalize test set using train norms
    x_cell_test_tensor = x_cell_test_tensor / cell_norms
    x_drug_test_tensor = x_drug_test_tensor / drug_norms
    
    # Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test), shuffle=False)

    # Test the model
    return test(model, test_loader)

def cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=2, visualize='first', run_id=0, split= 'LDO'):
    """
    split: 'LDO' (Leave-Drug-Out) or 'LCO' (Leave-Cell-Out)
    """
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], 'Balanced Accuracy':[], "Precision": [], "Recall": [], "F1 score": []}

    # Group for LDO/LCO (LDO: by drug ID or LCO: by cell ID)
    if split.upper() == 'LDO':
        groups = parse_drug_ids_from_pairs(x_cell_train.index)
    elif split.upper() == 'LCO':
        groups = parse_cell_ids_from_pairs(x_cell_train.index)
        
    # Labels as 0/1
    y_np = np.asarray(y_train).ravel()

    # Concatenate to split/resample by pair
    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)

    # StratifiedGroupKFold keep 0/1 proportion while separate by drug or cell line depending on the strategy
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_data, val_data) in enumerate(cv.split(np.zeros(len(y_np)), y_np, groups)):
        print(f"Fold {fold+1} ({split.upper()})")

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

        # Evaluate on the entire validation fold in a single pass
        val_loader_full = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        results = test(model, val_loader_full)
        
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

            # z_cell / z_drug
            tsne_labeled(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
            tsne_labeled(z_drug_fold, y_train_tensor, "t-SNE - z_drug")

            # concatenated: blue/red by labels
            tsne_labeled(torch.cat([z_cell_fold, z_drug_fold], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

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
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], 'Balanced Accuracy':[], "Precision": [], "Recall": [], "F1 score": []}
    
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
                                                              raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                              screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                              sep="\t")

        print('test_data when loaded:', test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")
        
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

        print('train_data after feature intersection with test set:', train_data.keys())
        for key, df in train_data.items():
            print(f"{key}: {df.shape}")

    # Prepare input data for training
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                   train_drug_screen)

    

    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)
    
    # Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        if is_test:

            # Train and evaluate the DeepDRA model on test data
            results = train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device)
            # Display final results
            Evaluation.add_results(history, results)

        else:

            results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5, visualize='first', run_id=i, split='LCO')
            if isinstance(results.get('AUC', None), list):
                for m in history:
                    history[m].extend(results[m])
            else:
                Evaluation.add_results(history, results)

    # Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=True)