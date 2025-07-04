from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from DeepDRA_clinical import DeepDRA, train, test
from data_loader_clinical import RawDataLoader
from evaluation import Evaluation
from utils import *
import random
import torch
import numpy as np
import pandas as pd

# Step 1: Define the batch size for training
batch_size = 64

# Step 2: Instantiate the combined model
ae_latent_dim = 50
num_epochs = 25

# Define the path to the clinical data file
clinical_file_path = 'C:/Users/camil/IMAG/DeepDRA/clinical/matched_cell_lines_annotations.tsv'

# Load the clinical dataset
clinical_df = RawDataLoader.load_clinical_data(clinical_file_path)

# Split the data into training and test sets
x_clinical_train, x_clinical_test = train_test_split(clinical_df, test_size=0.2, random_state=42)


clinical_input_dim = x_clinical_train.shape[1]

def train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, x_clinical_train, x_clinical_test, y_train, y_test, cell_sizes, drug_sizes, device):
    """

    Train and evaluate the DeepDRA model.

    Parameters:
    - X_cell_train (pd.DataFrame): Training data for the cell modality.
    - X_cell_test (pd.DataFrame): Test data for the cell modality.
    - X_drug_train (pd.DataFrame): Training data for the drug modality.
    - X_drug_test (pd.DataFrame): Test data for the drug modality.
    - X_clinical_train (pd.DataFrame): Clinical data corresponding to training samples.
    - y_train (array-like): Training labels.
    - y_test (array-like): Test labels.
    - cell_sizes (list): Sizes of the cell modality features.
    - drug_sizes (list): Sizes of the drug modality features.
    - device: PyTorch device.

    Returns:
    - result: Evaluation result on the test set.
    """

    model = DeepDRA(cell_sizes, drug_sizes, clinical_input_dim, ae_latent_dim, ae_latent_dim)
    model= model.to(device)

    print("x_cell_train.shape:", x_cell_train.shape)
    print("x_drug_train.shape:", x_drug_train.shape)
    print("x_clinical_train.shape:", x_clinical_train.shape)
    print("y_train.shape:", pd.Series(y_train).shape)

    # Step 3: Convert your training data to PyTorch tensors
    x_cell_train_tensor = torch.Tensor(x_cell_train.values)
    x_drug_train_tensor = torch.Tensor(x_drug_train.values)
    x_clinical_train_tensor = torch.Tensor(x_clinical_train.values)


    # Normaliser les tenseurs des modalités cell et drug 
    x_cell_train_tensor = torch.nn.functional.normalize(x_cell_train_tensor, dim=0)
    x_drug_train_tensor = torch.nn.functional.normalize(x_drug_train_tensor, dim=0)
    y_train_tensor = torch.Tensor(y_train)
    
    y_train_tensor = y_train_tensor.unsqueeze(1)

    # Compute class weights
    classes = np.array([0, 1])  # Assuming binary classification
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                 dtype=torch.float32)

    x_cell_train_tensor, x_cell_val_tensor, x_drug_train_tensor, x_drug_val_tensor, x_clinical_train_tensor, x_clinical_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        x_cell_train_tensor, x_drug_train_tensor, x_clinical_train_tensor, y_train_tensor, test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True)

    # Step 4: Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_cell_train_tensor.to(device), x_drug_train_tensor.to(device), x_clinical_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(x_cell_val_tensor.to(device), x_drug_val_tensor.to(device), x_clinical_val_tensor.to(device), y_val_tensor.to(device))
    
    # Step 5: Create the train_loader and val_loader
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
    x_clinical_test_tensor = torch.Tensor(x_clinical_test.values)
    y_test_tensor = torch.Tensor(y_test).to(device)

    # normalize data
    x_cell_test_tensor = torch.nn.functional.normalize(x_cell_test_tensor, dim=0).to(device)
    x_drug_test_tensor = torch.nn.functional.normalize(x_drug_test_tensor, dim=0).to(device)

    # Step 10: Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_cell_test_tensor, x_drug_test_tensor, x_clinical_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

    # Step 11: Test the model
    return test(model, test_loader)

# !!! MANQUE DES MODIF DANS CETTE DEF POUR QUE CV_TRAIN INTÈGRE LES DONNÉES CLINIQUES !!! 

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
        x_clinical_train_tensor = torch.Tensor(x_clinical_train.values)

        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)

        # Compute class weights
        classes = [0, 1]  # Assuming binary classification
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                     dtype=torch.float32)

        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_cell_train_tensor, x_drug_train_tensor, x_clinical_train_tensor, y_train_tensor)

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
    train_data, train_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                            raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                            screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                            sep="\t")
    
    # Step 3: Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=TCGA_DATA_FOLDER,
                                                              screen_file_directory=TCGA_SCREENING_DATA,
                                                              sep="\t")
        
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

        #common_columns = list(set(train_drug_screen.columns) & set(test_drug_screen.columns))
        
        #train_drug_screen.drop(common_columns[1:100], axis=1, inplace=True)
        #test_drug_screen = test_drug_screen[common_columns[1:100]]
    
    # Step 4: Prepare input data for training
    x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_data,
                                                                                                   train_drug_screen)

    # Chargez le fichier des annotations cliniques
    clinical_data = RawDataLoader.load_clinical_data('C:/Users/camil/IMAG/DeepDRA/clinical/matched_cell_lines_annotations.tsv', index_col='Cell_Line')
    
    # 1. Extraire le nom de la lignée cellulaire depuis l'index de x_cell_train (format "(CellLine,Drug)")
    extracted_names = x_cell_train.index.str.extract(r'\(([^,]+),')[0]
    
    # 2. Créer un masque pour retenir uniquement les échantillons dont le nom (extrait) est présent dans clinical_data
    mask = extracted_names.isin(clinical_data.index)
    
    # 3. Appliquer ce masque aux DataFrames omiques en utilisant mask.values pour l'indexation par position
    x_cell_train = x_cell_train[mask.values]
    x_drug_train = x_drug_train[mask.values]
    # Mettre à jour les index pour que chaque paire cell-drug reçoive le nom de la lignée
    x_cell_train.index = extracted_names[mask]
    x_drug_train.index = extracted_names[mask]
    # Pour y_train, convertir en Series et utiliser mask.values
    y_train = pd.Series(y_train, index=extracted_names.index)[mask.values].values


    if is_test:
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_data,
                                                                                                    test_drug_screen)
    
    # 4. Récupérer les données cliniques pour chaque lignée cellulaire
    clinical_x = clinical_data.loc[x_cell_train.index]
    
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    dataset = pd.concat([x_cell_train, x_drug_train, clinical_x], axis=1)
    dataset.index = x_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)

    x_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    x_drug_train = dataset.iloc[:, sum(cell_sizes):sum(cell_sizes) + sum(drug_sizes)]
    x_clinical_train = dataset.iloc[:, sum(cell_sizes) + sum(drug_sizes):]

    print("x_cell_train.shape:", x_cell_train.shape)
    print("x_drug_train.shape:", x_drug_train.shape)
    print("x_clinical_train.shape:", x_clinical_train.shape)

    
    # Step 5: Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        # Step 6: If is_test is True, perform random under-sampling on the training data
        if is_test:

            # Step 7: Train and evaluate the DeepDRA model on test data
            results = train_DeepDRA(x_cell_train, x_cell_test, x_drug_train, x_drug_test, x_clinical_train, x_clinical_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)

        else:
            # Step 8: Split the data into training and validation sets
            x_cell_train_split, x_cell_val, x_drug_train_split, x_drug_val, x_clinical_train_split, x_clinical_val, y_train_split, y_val = train_test_split(x_cell_train,
                                                                                                      x_drug_train, x_clinical_train, y_train,
                                                                                                      test_size=0.2,
                                                                                                      random_state=RANDOM_SEED,
                                                                                                      shuffle=True)
            # Step 9: Train and evaluate the DeepDRA model on the split data
            results = train_DeepDRA(x_cell_train_split, x_cell_val, x_drug_train_split, x_drug_val, x_clinical_train_split, x_clinical_val, y_train_split, y_val, cell_sizes,
                                     drug_sizes, device)

            #results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5)    # ! PAS ENCORE OPERATIONNEL !

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)

    # Step 11: Display final results
    Evaluation.show_final_results(history)
    return history

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(3, is_test=False)