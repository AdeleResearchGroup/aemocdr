#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


# In[2]:


from autoencoder import Autoencoder
from evaluation import Evaluation
from mlp import MLP
from utils import MODEL_FOLDER


# In[3]:


class DeepDRA(nn.Module):
    """
    DeepDRA (Deep Drug Response Anticipation) is a neural network model composed of two autoencoders for cell and drug modalities
    and an MLP for integrating the encoded features and making predictions.

    Parameters:
    - cell_modality_sizes (list): Sizes of the cell modality features.
    - drug_modality_sizes (list): Sizes of the drug modality features.
    - cell_ae_latent_dim (int): Latent dimension for the cell autoencoder.
    - drug_ae_latent_dim (int): Latent dimension for the drug autoencoder.
    - clinical_input_dim (int): Nombre de features cliniques.
    - mlp_input_dim (int): Input dimension for the MLP.
    - mlp_output_dim (int): Output dimension for the MLP.
    """

    def __init__(self, cell_modality_sizes, drug_modality_sizes, clinical_input_dim, cell_ae_latent_dim, drug_ae_latent_dim):
        super(DeepDRA, self).__init__()

        # Initialize cell and drug autoencoders
        self.cell_autoencoder = Autoencoder(sum(cell_modality_sizes), cell_ae_latent_dim)
        self.drug_autoencoder = Autoencoder(sum(drug_modality_sizes), drug_ae_latent_dim)

        # Store modality sizes
        self.cell_modality_sizes = cell_modality_sizes
        self.drug_modality_sizes = drug_modality_sizes

        # Calcul de la dimension d'entrée du MLP : concaténation des 3 latent spaces
        mlp_input_dim = cell_ae_latent_dim + drug_ae_latent_dim + clinical_input_dim

        # Initialize MLP
        self.mlp = MLP(mlp_input_dim, 1)

    def forward(self, cell_x, drug_x, clinical_x):
        """
        Forward pass of the DeepDRA model including la branche clinique.

        Parameters:
        - cell_x (torch.Tensor): Input tensor for cell modality.
        - drug_x (torch.Tensor): Input tensor for drug modality.

        Returns:
        - cell_decoded (torch.Tensor): Decoded tensor for the cell modality.
        - drug_decoded (torch.Tensor): Decoded tensor for the drug modality.
        - mlp_output (torch.Tensor): Output tensor from the MLP.
        """
        # Encode and decode cell modality
        cell_encoded = self.cell_autoencoder.encoder(cell_x)
        cell_decoded = self.cell_autoencoder.decoder(cell_encoded)

        # Encode and decode drug modality
        drug_encoded = self.drug_autoencoder.encoder(drug_x)
        drug_decoded = self.drug_autoencoder.decoder(drug_encoded)

        # Fusion des représentations latentes
        mlp_output = self.mlp(torch.cat((cell_encoded, drug_encoded, clinical_x), 1))

        return cell_decoded, drug_decoded, mlp_output

    def compute_l1_loss(self, w):
        """
        Computes L1 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L1 regularization loss.
        """
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        """
        Computes L2 regularization loss.

        Parameters:
        - w (torch.Tensor): Input tensor.

        Returns:
        - loss (torch.Tensor): L2 regularization loss.
        """
        return torch.square(w).sum()


def train(model, train_loader, val_loader,  num_epochs,class_weights):
    """
    Trains the DeepDRA (Deep Drug Response Anticipation) model.

    Parameters:
    - model (DeepDRA): The DeepDRA model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - num_epochs (int): Number of training epochs.
    """


    autoencoder_loss_fn = nn.MSELoss()
    mlp_loss_fn = nn.BCELoss()

    train_accuracies = []
    val_accuracies = []

    train_loss = []
    val_loss = []

    mlp_optimizer = optim.Adam(model.parameters(), lr=0.0005,)
    scheduler = lr_scheduler.ReduceLROnPlateau(mlp_optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    # Define weight parameters for each loss term
    cell_ae_weight = 1.0
    drug_ae_weight = 1.0
    mlp_weight = 1.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total_samples = 0
        for batch_idx, (cell_data, drug_data, clinical_data_batch, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()

            # Forward pass
            cell_decoded_output, drug_decoded_output, mlp_output = model(cell_data, drug_data, clinical_data_batch)

            # Compute class weights for the current batch
            # batch_class_weights = class_weights[target.long()]
            # mlp_loss_fn = nn.BCEWithLogitsLoss(weight=batch_class_weights)

            # Compute losses
            cell_ae_loss = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output, cell_data)
            drug_ae_loss = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output, drug_data)
            mlp_loss = mlp_weight * mlp_loss_fn(mlp_output, target)

            # Total loss is the sum of autoencoder losses and MLP loss
            total_loss = drug_ae_loss + cell_ae_loss + mlp_loss

            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            mlp_optimizer.step()
            total_train_loss += total_loss.item()

            # Calculate accuracy
            train_predictions = torch.round(mlp_output)
            train_correct += (train_predictions == target).sum().item()
            train_total_samples += target.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch_idx, (cell_data_val, drug_data_val, clinical_data_val, val_target) in enumerate(val_loader):
                cell_decoded_output_val, drug_decoded_output_val, mlp_output_val = model(cell_data_val, drug_data_val, clinical_data_val)
                # batch_class_weights = class_weights[val_target.long()]
                # mlp_loss_fn = nn.BCEWithLogitsLoss(weight=batch_class_weights)

                # Compute losses
                cell_ae_loss_val = cell_ae_weight * autoencoder_loss_fn(cell_decoded_output_val, cell_data_val)
                drug_ae_loss_val = drug_ae_weight * autoencoder_loss_fn(drug_decoded_output_val, drug_data_val)
                mlp_loss_val = mlp_weight * mlp_loss_fn(mlp_output_val, val_target)

                # Total loss is the sum of autoencoder losses and MLP loss
                total_val_loss = (drug_ae_loss_val + cell_ae_loss_val + mlp_loss_val).item()

                # Calculate accuracy
                val_predictions = torch.round(mlp_output_val)
                correct += (val_predictions == val_target).sum().item()
                total_samples += val_target.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)


        train_accuracy = train_correct / train_total_samples
        train_accuracies.append(train_accuracy)
        val_accuracy = correct / total_samples
        val_accuracies.append(val_accuracy)

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}'.format(
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, train_accuracy,
                val_accuracy))

        # Learning rate scheduler step
        scheduler.step(total_train_loss)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_FOLDER + 'DeepDRA.pth')


def test(model, test_loader, reverse=False):
    """
    Tests the given model on the test dataset using evaluation metrics.

    Parameters:
    - model: The trained model to be evaluated.
    - test_loader: DataLoader for the test dataset.
    - reverse (bool): If True, reverse the predictions for evaluation.

    Returns:
    - result: The evaluation result based on the chosen metrics.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_labels = []

    # Iterate over the test dataset
    for i, (test_cell_loader, test_drug_loader, test_clinical_loader, labels) in enumerate(test_loader):
        # Forward pass through the model
        with torch.no_grad():
            decoded_cell_output, decoded_drug_output, mlp_output = model(test_cell_loader, test_drug_loader, test_clinical_loader)

        # Apply reverse if specified
        predictions = 1 - mlp_output if reverse else mlp_output

    # Evaluate the predictions using the specified metrics
    result = Evaluation.evaluate(labels, predictions)

    return result


# In[ ]:




