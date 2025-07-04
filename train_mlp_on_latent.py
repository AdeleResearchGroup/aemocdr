# =============================
# Script 2 : train_mlp_on_latent.py + save AEs and MLP
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from data_loader_pretraining import RawDataLoader
from utils import DATA_MODALITIES, RAW_BOTH_DATA_FOLDER, BOTH_SCREENING_DATA_FOLDER, CCLE_RAW_DATA_FOLDER, CCLE_SCREENING_DATA_FOLDER, TCGA_DATA_FOLDER, TCGA_SCREENING_DATA
from mlp import MLP
from evaluation import Evaluation

# À choisir ci-dessous quel type de script utiliser : SimpleAutoencoder (MSE LOSS) vs. ZINBAutoencoder (ZINB loss)
#from pretrain_autoencoders import SimpleAutoencoder
from pretrain_autoencoders_ZINB import ZINBAutoencoder

# Utiliser le GPU si disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

RANDOM_SEED = 42

class DeepDRA_pretrained(nn.Module):

    def __init__(self, encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False):
        super(DeepDRA_pretrained, self).__init__()

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

def DeepDRA_pretrained_training(x_cell_train, x_cell_val, x_drug_train, x_drug_val, y_train, y_val, run_id=None, visualize='first'):

    cell_ae_latent_dim = 700
    drug_ae_latent_dim = 50
    batch_size = 64

    # Convert DataFrames to Pytorch Tensors
    x_cell_train_tensor = torch.Tensor(x_cell_train.values)
    x_drug_train_tensor = torch.Tensor(x_drug_train.values)
    
    x_cell_train_tensor = torch.nn.functional.normalize(x_cell_train_tensor, dim=0)
    x_drug_train_tensor = torch.nn.functional.normalize(x_drug_train_tensor, dim=0)
    
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = y_train_tensor.unsqueeze(1)

    # Compute class weights
    classes = np.array([0, 1])  # Assuming binary classification
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                 dtype=torch.float32)

    x_cell_train_tensor, x_cell_val_tensor, x_drug_train_tensor, x_drug_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
    x_cell_train_tensor, x_drug_train_tensor, y_train_tensor, test_size=0.1,
    random_state=RANDOM_SEED,
    shuffle=True)
    
    encoder_cell = ZINBAutoencoder(x_cell_train_tensor.shape[1], cell_ae_latent_dim)
    encoder_cell.load_state_dict(torch.load("encoder_cell.pth"))
    
    encoder_drug = ZINBAutoencoder(x_drug_train_tensor.shape[1], drug_ae_latent_dim)
    encoder_drug.load_state_dict(torch.load("encoder_drug.pth"))
    
    model = DeepDRA_pretrained(encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False)
    model= model.to(device)

    # Affichage du modèle
    if run_id==0:
        print("\nArchitecture du modèle:\n")
        print(model)
    
    # Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor.to(device), x_drug_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(x_cell_val_tensor.to(device), x_drug_val_tensor.to(device), y_val_tensor.to(device))

    # Create the train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    train_mlp_with_encoders(model, train_loader, val_loader, class_weights, run_id=run_id)
  

    # sparcity
    with torch.no_grad():
        z_cell = encoder_cell.encoder(x_cell_train_tensor)
        z_drug = encoder_drug.encoder(x_drug_train_tensor)
        sparsity_cell = (z_cell == 0).float().mean().item()
        sparsity_drug = (z_drug == 0).float().mean().item()
        print(f"Sparsity z_cell: {sparsity_cell:.4f}, z_drug: {sparsity_drug:.4f}")

    # T-SNE (never / first / always)
    if visualize == 'always' or (visualize == 'first' and run_id == 0):
        def plot_tsne(z_tensor, y_tensor, title):
            z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.cpu().numpy())
            y_np = y_tensor.cpu().numpy().ravel()
            plt.figure(figsize=(8, 6))
            for label, color in zip([0, 1], ['blue', 'red']):
                plt.scatter(z_embedded[y_np == label, 0], z_embedded[y_np == label, 1],
                            label='Resistant' if label == 0 else 'Sensitive',
                            c=color, alpha=0.7)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_tsne(z_cell, y_train_tensor, "t-SNE - z_cell")
        plot_tsne(z_drug, y_train_tensor, "t-SNE - z_drug")
        plot_tsne(torch.cat([z_cell, z_drug], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    return model

def train_mlp_with_encoders(model, train_loader, val_loader, class_weights, epochs=25, run_id=0):

    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    mlp_loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_preds = []
        train_targets = []
        for batch_idx, (cell_data, drug_data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()
            y_pred, _, _ = model(cell_data, drug_data)
            loss = mlp_loss_fn(y_pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            mlp_optimizer.step()
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
                val_loss = mlp_loss_fn(y_val_pred, val_target)
                total_val_loss += val_loss.item()

                val_preds.extend((y_val_pred > 0.5).cpu().numpy())
                val_targets.extend(val_target.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_accuracies.append(val_acc)

        train_losses.append(total_train_loss / len(train_loader))
        val_losses.append(total_val_loss / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        scheduler.step(total_val_loss / len(val_loader))

    # Plot losses and accuracies
    Evaluation.plot_train_val_loss(train_losses, val_losses, epochs)
    Evaluation.plot_train_val_accuracy(train_accuracies, val_accuracies, epochs)

    # Save each models
    torch.save(model.encoder_cell, f"encoder_cell_trained_run_{run_id}.pt")
    torch.save(model.encoder_drug, f"encoder_drug_trained_run_{run_id}.pt")
    torch.save(model.mlp.state_dict(), f"mlp_trained_run_{run_id}.pth")
    
    return model

def test(model, test_loader):
    model.eval()
    
    for test_cell_loader, test_drug_loader, labels in test_loader:
        # Forward pass through the model
        with torch.no_grad():
            y_pred, _, _ = model(test_cell_loader, test_drug_loader)

    result = Evaluation.evaluate(labels, y_pred)
    return result

def cv_train(x_cell_train, x_drug_train, y_train, device, k=5, run_id=None, visualize='first'):


    splits = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}
    
    cell_ae_latent_dim = 700
    drug_ae_latent_dim = 25
    batch_size = 64

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(x_cell_train)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

            # Convert your training data to PyTorch tensors
        x_cell_train_tensor = torch.Tensor(x_cell_train.values)
        x_drug_train_tensor = torch.Tensor(x_drug_train.values)

        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)
        
        x_cell_train_tensor = torch.nn.functional.normalize(x_cell_train_tensor, dim=0)
        x_drug_train_tensor = torch.nn.functional.normalize(x_drug_train_tensor, dim=0)

        # Compute class weights
        classes = np.array([0, 1])  # Assuming binary classification
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                     dtype=torch.float32)

        encoder_cell = ZINBAutoencoder(x_cell_train_tensor.shape[1], cell_ae_latent_dim)
        encoder_cell.load_state_dict(torch.load("encoder_cell.pth"))
        
        encoder_drug = ZINBAutoencoder(x_drug_train_tensor.shape[1], drug_ae_latent_dim)
        encoder_drug.load_state_dict(torch.load("encoder_drug.pth"))
        
        model = DeepDRA_pretrained(encoder_cell, encoder_drug, cell_ae_latent_dim, drug_ae_latent_dim, freeze_encoders=False)
        model = model.to(device)
      
        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_cell_train_tensor.to(device), x_drug_train_tensor.to(device), y_train_tensor.to(device))

        # Create the train_loader AND val_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
        
        # Train the model
        train_mlp_with_encoders(model, train_loader, val_loader, class_weights)

        # Create a TensorDataset with the input features and target labels
        test_loader = DataLoader(train_dataset, batch_size=len(x_cell_train), sampler=test_sampler)

        # Test the model
        results = test(model, test_loader)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)


        # sparcity
        with torch.no_grad():
            z_cell = encoder_cell.encoder(x_cell_train_tensor)
            z_drug = encoder_drug.encoder(x_drug_train_tensor)
            sparsity_cell = (z_cell == 0).float().mean().item()
            sparsity_drug = (z_drug == 0).float().mean().item()
            print(f"Sparsity z_cell: {sparsity_cell:.4f}, z_drug: {sparsity_drug:.4f}")
        
        # T-SNE (never / first / always)
        if visualize == 'always' or (visualize == 'first' and run_id == 0):
            def plot_tsne(z_tensor, y_tensor, title):
                z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_tensor.cpu().numpy())
                y_np = y_tensor.cpu().numpy().ravel()
                plt.figure(figsize=(8, 6))
                for label, color in zip([0, 1], ['blue', 'red']):
                    plt.scatter(z_embedded[y_np == label, 0], z_embedded[y_np == label, 1],
                                label='Resistant' if label == 0 else 'Sensitive',
                                c=color, alpha=0.7)
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.show()
    
            plot_tsne(z_cell, y_train_tensor, "t-SNE - z_cell")
            plot_tsne(z_drug, y_train_tensor, "t-SNE - z_drug")
            plot_tsne(torch.cat([z_cell, z_drug], dim=1), y_train_tensor, "t-SNE - z_cell + z_drug")

    return Evaluation.show_final_results(history)


def run(k=10, is_test=False, visualize='first'):
    
    # Initialisation de l'historique des métriques
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}
    
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
            raw_file_directory=TCGA_DATA_FOLDER,  
            screen_file_directory=TCGA_SCREENING_DATA,
            sep="\t"
        )

        print('test_data when loaded:', test_data.keys())
        for key, df in test_data.items():
            print(f"{key}: {df.shape}")
                
        # Intersection des features entre train et test
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

    # Application d'un sous-échantillonnage pour équilibrer la classe majoritaire    
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    dataset = pd.concat([x_cell_train, x_drug_train], axis=1)
    dataset.index = x_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)
    x_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    x_drug_train = dataset.iloc[:, sum(cell_sizes):]

    print('x_cell_train shape:', x_cell_train.shape)
    print('x_drug_train shape:', x_cell_train.shape)

    from collections import Counter
    print("Distribution des classes :", Counter(y_train))

    for i in range(k):
        print(f"\nRun {i+1}/{k}")

        # If is_test is True, perform random under-sampling on the training data
        if is_test:
            
            # Train and evaluate the DeepDRA model on test data
            model = DeepDRA_pretrained_training(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, run_id=i, visualize=visualize)
            
            # Convert your test data to PyTorch tensors
            x_cell_test_tensor = torch.Tensor(x_cell_test.values)
            x_drug_test_tensor = torch.Tensor(x_drug_test.values)
            y_test_tensor = torch.Tensor(y_test).to(device)
        
            # normalize data
            x_cell_test_tensor = torch.nn.functional.normalize(x_cell_test_tensor, dim=0).to(device)
            x_drug_test_tensor = torch.nn.functional.normalize(x_drug_test_tensor, dim=0).to(device)
        
            # Create a TensorDataset with the input features and target labels for testing
            test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

            results = test(model, test_loader)

        else:

            # Train and evaluate the DeepDRA model on the split data
            results = cv_train(x_cell_train, x_drug_train, y_train, device, run_id=i, visualize=visualize)

        # Ajout des métriques du run courant à l'historique
        Evaluation.add_results(history, results)

    # Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(k=5, is_test=True)