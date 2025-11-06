# =============================
# Script 2 : train_mlp_on_latent.py - LO strategy
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold
from data_loader_pretraining import RawDataLoader
from utils import DATA_MODALITIES, RAW_BOTH_DATA_FOLDER, BOTH_SCREENING_DATA_FOLDER, CCLE_RAW_DATA_FOLDER, CCLE_SCREENING_DATA_FOLDER, TCGA_DATA_FOLDER, TCGA_SCREENING_DATA
from mlp import MLP
from evaluation import Evaluation

# Choose below which type of script to use: SimpleAutoencoder (MSE LOSS) vs. ZINBAutoencoder (ZINB loss)
from pretrain_autoencoders import SimpleAutoencoder
#from pretrain_autoencoders_ZINB import ZINBAutoencoder

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

RANDOM_SEED = 42

class TriMORDR_pretrained(nn.Module):

    def __init__(self, encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False):
        super(TriMORDR_pretrained, self).__init__()

        # Load pretrained encoders
        self.encoder_cell = encoder_cell.encoder
        self.encoder_drug = encoder_drug.encoder

        if freeze_encoders:
            for param in self.encoder_cell.parameters():
                param.requires_grad = False
            for param in self.encoder_drug.parameters():
                param.requires_grad = False
                
        # Initialize MLP
        self.mlp = MLP(cell_ae_latent_dim+drug_ae_latent_dim, 1)


    def forward(self, cell_x, drug_x):
        z_cell = self.encoder_cell(cell_x)
        z_drug = self.encoder_drug(drug_x)

        combined = torch.cat([z_cell, z_drug], dim=1)
        return self.mlp(combined), z_cell, z_drug

cell_ae_latent_dim = 700
drug_ae_latent_dim = 50
batch_size = 64
num_epochs = 25

def parse_cell_ids_from_pairs(index_like):
    s = pd.Index(index_like).to_series()
    # capture tout avant la 1ère virgule: "(CELL,DRUG)" -> "CELL"
    return s.str.extract(r'^\(([^,]+),')[0].values

def parse_drug_ids_from_pairs(index_like):
    s = pd.Index(index_like).to_series()
    # capture tout après la virgule: "(CELL,DRUG)" -> "DRUG"
    return s.str.extract(r'^\([^,]+,\s*([^)]+)\)')[0].values

def TriMORDR_pretrained_training(x_cell_train, x_drug_train, y_train, run_id=None, visualize='first', split='LCO'):
    """
    Split 90/10 inside, compute normalization factors on the RAW train split (BEFORE RUS),
    apply RUS on the train split only, then reuse the SAME factors for train-RUS and val
    Returns: model, cell_norms, drug_norms
    """
    
    # Group for LDO/LCO (LDO: by drug ID or LCO: by cell ID)
    if split.upper() == 'LDO':
        groups = parse_drug_ids_from_pairs(x_cell_train.index)
    elif split.upper() == 'LCO':
        groups = parse_cell_ids_from_pairs(x_cell_train.index)

    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    n_cell = x_cell_train.shape[1] 
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
    
    # RUS on training set
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
    
    # Convert to PyTorch tensors
    x_cell_train_tensor = torch.Tensor(x_cell_train.values) / cell_norms
    x_drug_train_tensor = torch.Tensor(x_drug_train.values) / drug_norms 

    x_cell_val_tensor = torch.Tensor(x_cell_val.values) / cell_norms
    x_drug_val_tensor = torch.Tensor(x_drug_val.values) / drug_norms 

    y_train_tensor = torch.Tensor(y_train).unsqueeze(1)
    y_val_tensor = torch.Tensor(y_val).unsqueeze(1)
    
    # Send to device
    x_cell_train_tensor = x_cell_train_tensor.to(device); x_drug_train_tensor = x_drug_train_tensor.to(device); y_train_tensor = y_train_tensor.to(device)
    x_cell_val_tensor = x_cell_val_tensor.to(device); x_drug_val_tensor = x_drug_val_tensor.to(device); y_val_tensor = y_val_tensor.to(device)

    # Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_cell_val_tensor, x_drug_val_tensor, y_val_tensor)

    # Create the train_loader and val_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_drug  = x_drug_train.shape[1]

    # Pretrained encoders
    encoder_cell = SimpleAutoencoder(x_cell_train_tensor.shape[1], cell_ae_latent_dim)
    encoder_cell.load_state_dict(torch.load("encoder_cell.pth"))
    
    encoder_drug = SimpleAutoencoder(x_drug_train_tensor.shape[1], drug_ae_latent_dim)
    encoder_drug.load_state_dict(torch.load("encoder_drug.pth"))
    
    model = TriMORDR_pretrained(encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False).to(device)

    # Display model
    if run_id==0:
        print("\nModel architecture:\n")
        print(model)
    
    # Train the model
    train_mlp_with_encoders(model, train_loader, val_loader, num_epochs, run_id=run_id)

    # T-SNE (never / first / always)
    if visualize == 'always' or (visualize == 'first' and run_id == 0):
        with torch.no_grad():
            model.encoder_cell.eval(); model.encoder_drug.eval()
            z_cell = model.encoder_cell(x_cell_train_tensor)
            z_drug = model.encoder_drug(x_drug_train_tensor)
    
            def plot_tsne(z_tensor, y_tensor, title):
                z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.cpu().numpy())
                y_np = y_tensor.cpu().numpy().ravel()
                plt.figure(figsize=(8, 6))
                for label, color in zip([0, 1], ['blue', 'red']):
                    plt.scatter(z_embedded[y_np == label, 0], z_embedded[y_np == label, 1],
                                label='Resistant' if label == 0 else 'Sensitive',
                                c=color, s=10, alpha=0.7)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.show()
    
            plot_tsne(z_cell, y_train_tensor, "t-SNE - z_cell")
            plot_tsne(z_drug, y_train_tensor, "t-SNE - z_drug")
            plot_tsne(torch.cat([z_cell, z_drug], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    return model, cell_norms.detach().cpu(), drug_norms.detach().cpu()

def train_mlp_with_encoders(model, train_loader, val_loader, num_epochs, run_id=0):
    
    # Optimizer: different learning rate (encoders vs MLP)
    enc_lr = 1e-3
    mlp_lr = 5e-4
    optimizer = optim.Adam([
    {'params': model.encoder_cell.parameters(), 'lr': enc_lr, 'weight_decay': 1e-5},
    {'params': model.encoder_drug.parameters(), 'lr': enc_lr, 'weight_decay': 1e-5},
    {'params': model.mlp.parameters(),         'lr': mlp_lr, 'weight_decay': 1e-5},
])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_preds = []
        train_targets = []
        for batch_idx, (cell_data, drug_data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred, _, _ = model(cell_data, drug_data)
            loss = loss_fn(y_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_train_loss += loss.item()

            # Collect predictions and targets for accuracy
            train_preds.extend((y_pred > 0.5).cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for val_batch_idx, (cell_data_val, drug_data_val, val_target) in enumerate(val_loader):
                y_val_pred, _, _ = model(cell_data_val, drug_data_val)
                val_loss = loss_fn(y_val_pred, val_target)
                total_val_loss += val_loss.item()

                val_preds.extend((y_val_pred > 0.5).cpu().numpy())
                val_targets.extend(val_target.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_accuracies.append(val_acc)

        train_losses.append(total_train_loss / len(train_loader))
        val_losses.append(total_val_loss / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        scheduler.step(total_val_loss / len(val_loader))

    # Plot losses and accuracies
    Evaluation.plot_train_val_loss(train_losses, val_losses, num_epochs)
    Evaluation.plot_train_val_accuracy(train_accuracies, val_accuracies, num_epochs)

    # Save each models
    torch.save(model.encoder_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(model.encoder_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(model.mlp.state_dict(), f"mlp_trained_run_{run_id}.pth")
    
    return model

def test(model, test_loader):
    model.eval()
    
    preds_all = []
    labels_all = []    
    with torch.no_grad():
        for xc, xd, y in test_loader:
        # Forward pass through the model
            y_pred, _, _ = model(xc, xd)
            preds_all.append(y_pred.detach().cpu())
            labels_all.append(y.detach().cpu())

    # Concat all batches
    y_pred_all = torch.cat(preds_all, dim=0)
    y_true_all = torch.cat(labels_all, dim=0)
    
    result = Evaluation.evaluate(y_true_all, y_pred_all)
    return result

def cv_train(x_cell_train, x_drug_train, y_train, device, k=5, run_id=None, visualize='first', split= 'LDO'):
    """
    split: 'LDO' (Leave-Drug-Out) or 'LCO' (Leave-Cell-Out)
    """

    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], 'Balanced Accuracy':[], "Precision": [], "Recall": [], "F1 score": []}
    
    cell_ae_latent_dim = 700
    drug_ae_latent_dim = 50
    batch_size = 64

    # Group for LDO/LCO (LDO: by drug ID or LCO: by cell ID)
    if split.upper() == 'LDO':
        groups = parse_drug_ids_from_pairs(x_cell_train.index)
    elif split.upper() == 'LCO':
        groups = parse_cell_ids_from_pairs(x_cell_train.index)
        
    # Labels as 0/1
    y_np = np.asarray(y_train).ravel().astype(int)
    
    # Concatenate to split/resample by pair
    X_pairs = pd.concat([x_cell_train, x_drug_train], axis=1)
    n_cell = x_cell_train.shape[1]

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
        
        n_drug  = x_drug_train.shape[1]
        
        # Pretrained encoders
        encoder_cell = SimpleAutoencoder(n_cell, cell_ae_latent_dim)
        encoder_cell.load_state_dict(torch.load("encoder_cell.pth"))
        
        encoder_drug = SimpleAutoencoder(n_drug, drug_ae_latent_dim)
        encoder_drug.load_state_dict(torch.load("encoder_drug.pth"))
        
        model = TriMORDR_pretrained(encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False).to(device)

        # Display model
        if fold==0:
            print("\nArchitecture du modèle:\n")
            print(model)

        # Train model
        train_mlp_with_encoders(model, train_loader, val_loader, num_epochs, run_id=run_id)
      
        # Evaluate on the entire validation fold in a single pass
        val_loader_full = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        results = test(model, val_loader_full)

        # Add results to the history dictionary
        Evaluation.add_results(history, results)

        # T-SNE (never / first / always)
        if visualize == 'always' or (visualize == 'first' and fold == 0):
            with torch.no_grad():
                model.encoder_cell.eval()
                model.encoder_drug.eval()
                z_cell_fold = model.encoder_cell(x_cell_train_tensor)
                z_drug_fold = model.encoder_drug(x_drug_train_tensor)
            
                def plot_tsne(z_tensor, y_tensor, title):
                    z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.detach().cpu().numpy())
                    y_np = y_tensor.detach().cpu().numpy().ravel()
                    plt.figure(figsize=(8, 6))
                    for label, color in zip([0, 1], ['blue', 'red']):
                        plt.scatter(z_embedded[y_np == label, 0], z_embedded[y_np == label, 1],
                                    label='Resistant' if label == 0 else 'Sensitive',
                                    c=color, s=10, alpha=0.7)
                    plt.title(title)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
        
                plot_tsne(z_cell_fold, y_train_tensor, "t-SNE - z_cell")
                plot_tsne(z_drug_fold, y_train_tensor, "t-SNE - z_drug")
                plot_tsne(torch.cat([z_cell_fold, z_drug_fold], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    return history

def run(k=10, is_test=False, visualize='first'):
    
    # Initialization of metrics history
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], 'Balanced Accuracy':[], "Precision": [], "Recall": [], "F1 score": []}
    
    # Load training data
    train_data, train_drug_screen = RawDataLoader.load_data(
        data_modalities=DATA_MODALITIES,
        raw_file_directory=RAW_BOTH_DATA_FOLDER,
        screen_file_directory=BOTH_SCREENING_DATA_FOLDER,
        sep="\t"
    )

    print('train_data when loaded:', train_data.keys())
    for key, df in train_data.items():
        print(f"{key}: {df.shape}")
    
    # Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(
            data_modalities=DATA_MODALITIES,
            raw_file_directory=CCLE_RAW_DATA_FOLDER,  
            screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
            sep="\t"
        )

        print('test_data when loaded:', test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")
                
        # Intersection of features between train and test
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

        # Save the feature columns for reproducibility:
        all_features = {}
        for key, df in train_data.items():
            all_features[key] = df.columns.tolist()
        
        import pickle
        with open("feature_columns.pkl", "wb") as f:
            pickle.dump(all_features, f)

    # Prepare input data for training
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data, train_drug_screen)

    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data, test_drug_screen)


    for i in range(k):
        print(f"\nRun {i+1}/{k}")

        if is_test:
            
            # Train and evaluate the TriMORDR model on test data
            model, cell_norms, drug_norms = TriMORDR_pretrained_training(x_cell_train, x_drug_train, y_train, run_id=i, visualize=visualize, split='LCO')

            print(f"x_cell_test shape:  {x_cell_test.shape}")
            print(f"x_drug_test shape:  {x_drug_test.shape}")
            
            # Convert test data to PyTorch tensors
            x_cell_test_tensor = torch.Tensor(x_cell_test.values).to(device)
            x_drug_test_tensor = torch.Tensor(x_drug_test.values).to(device)
            y_test_tensor = torch.Tensor(y_test).to(device)

            # normalize test set using train norms
            x_cell_test_tensor = x_cell_test_tensor / cell_norms
            x_drug_test_tensor = x_drug_test_tensor / drug_norms
            
            # Create a TensorDataset with the input features and target labels for testing
            test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test), shuffle=False)

            results = test(model, test_loader)
                
            # Add the current run metrics to the history
            Evaluation.add_results(history, results)

        else:

            # Train and evaluate the TriMORDR model on the split data
            results = cv_train(x_cell_train, x_drug_train, y_train, device, k=5, run_id=i, visualize='first', split='LCO')

            if isinstance(results.get('AUC', None), list):
                for m in history:
                    history[m].extend(results[m])
            else:
                Evaluation.add_results(history, results)

    # Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(k=1, is_test=False)