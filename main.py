from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from DeepDRA_pretraining import DeepDRA, load_pretrained_encoder, train, test
from data_loader_pretraining import RawDataLoader
from evaluation import Evaluation
from utils import *
from AE_pretraining import SimpleAutoencoder
from MLP_pretraining import MLP
import random
import torch
import numpy as np
import pandas as pd

# Step 1: Define the batch size for training
batch_size = 64

# Step 2: Instantiate the combined model
ae_latent_dim = 50
num_epochs = 25

def train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device):
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
    model = DeepDRA(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim)
    model= model.to(device)

    # Step 3: Convert your training data to PyTorch tensors
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


    # Step 4: Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor.to(device), x_drug_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(x_cell_val_tensor.to(device), x_drug_val_tensor.to(device), y_val_tensor.to(device))
    
    # Step 5: Create the train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Step 6: Train the model
    train(model, train_loader, val_loader, num_epochs,class_weights)

    # Step 7: Save the trained model
    torch.save(model, 'DeepDRA.pth')

    # Step 8: Load the saved model
    model = torch.load('DeepDRA.pth', weights_only = False)

    # Step 9: Convert your test data to PyTorch tensors
    x_cell_test_tensor = torch.Tensor(x_cell_test.values)
    x_drug_test_tensor = torch.Tensor(x_drug_test.values)
    y_test_tensor = torch.Tensor(y_test).to(device)

    # normalize data
    x_cell_test_tensor = torch.nn.functional.normalize(x_cell_test_tensor, dim=0).to(device)
    x_drug_test_tensor = torch.nn.functional.normalize(x_drug_test_tensor, dim=0).to(device)

    # Step 10: Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

    # Step 11: Test the model
    return test(model, test_loader)

def cv_train(x_cell_train, x_drug_train, y_train, cell_sizes,
                                    drug_sizes, device, k=5, ):


    splits = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(x_cell_train)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        model = DeepDRA(cell_sizes, drug_sizes, ae_latent_dim, ae_latent_dim)
        # Convert your training data to PyTorch tensors
        x_cell_train_tensor = torch.Tensor(x_cell_train.values)
        x_drug_train_tensor = torch.Tensor(x_drug_train.values)

        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)

        # Compute class weights
        classes = [0, 1]  # Assuming binary classification
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                     dtype=torch.float32)

        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, y_train_tensor)

        # Create the train_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        # Train the model
        train(model, train_loader,train_loader, num_epochs, class_weights)


        # Create a TensorDataset with the input features and target labels
        test_loader = DataLoader(train_dataset, batch_size=len(x_cell_train), sampler=test_sampler)

        # Test the model
        results = test(model, test_loader)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)


    return Evaluation.show_final_results(history)

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
    # Step 1: Initialize a dictionary to store evaluation metrics
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}
    
    # Step 2: Load training data
    data_dict, _ = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                            raw_file_directory=RAW_BOTH_DATA_FOLDER,
                                            screen_file_directory=BOTH_SCREENING_DATA_FOLDER,
                                            sep="\t")

    # 2. Extraire X_cell_full et X_drug_full
    X_cell_full, X_drug_full = RawDataLoader.get_unique_entities(data_dict)

    # Préentraîner et sauvegarder
    train_autoencoder(X_cell_full, latent_dim=50, save_path="encoder_cell.pth", epochs=50)
    train_autoencoder(X_drug_full, latent_dim=50, save_path="encoder_drug.pth", epochs=50)
    
    # Step 3: Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                              screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                              sep="\t")
        
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

        #common_columns = list(set(train_drug_screen.columns) & set(test_drug_screen.columns))
        
        #train_drug_screen.drop(common_columns[1:100], axis=1, inplace=True)
        #test_drug_screen = test_drug_screen[common_columns[1:100]]
    
    # Step 4: Prepare input data for training
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                   train_drug_screen)


    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)

        
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    dataset = pd.concat([x_cell_train, x_drug_train], axis=1)
    dataset.index = x_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)
    x_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    x_drug_train = dataset.iloc[:, sum(cell_sizes):]

    # Step 5: Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        # Step 6: If is_test is True, perform random under-sampling on the training data
        if is_test:

            # Step 7: Train and evaluate the DeepDRA model on test data
            results = train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)

        else:
             # Step 8: Split the data into training and validation sets
             x_cell_train_split, x_cell_val, x_drug_train_split, x_drug_val, y_train_split, y_val = train_test_split(x_cell_train,
                                                                                                      x_drug_train, y_train,
                                                                                                      test_size=0.2,
                                                                                                      random_state=RANDOM_SEED,
                                                                                                      shuffle=True)
             # Step 9: Train and evaluate the DeepDRA model on the split data
             results = train_DeepDRA(x_cell_train_split, x_cell_val, x_drug_train_split, x_drug_val, y_train_split, y_val, cell_sizes, drug_sizes, device)

             #results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)

    # Step 11: Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=True)