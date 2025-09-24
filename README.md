Description of Scripts and Execution Order – 

I- AE‑MLP (DeepDRA‑derived) for Drug‑Response Prediction

A clean, ready‑to‑run baseline that learns separate cell and drug embeddings with autoencoders and predicts sensitivity (binary) with an MLP. The code is adapted from DeepDRA and organized for CTRP-GDSC → CCLE/TCGA style experiments.

AE-MLP_main.py — standard train/val or train→test runs.

AE-MLP_main_LO.py — “Leave-Out” splits (Leave-Drug-Out / Leave-Cell-Out) using grouped CV

	Steps:
		1 - In the utils.py file: Write the data modalities you want to use in this line: DATA_MODALITIES = ['cell_CN','cell_exp','cell_methy','cell_mut','drug_desc', 'drug_finger', 'drug_DT']

		2 - In the run() function, load the training data and if applicable the test data of choice by filling up these lines: raw_file_directory='dataset chosen'and screen_file_directory='corresponding screening file'
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)
			Corresponding screening files for each datasets: 'BOTH_SCREENING_DATA_FOLDER' (CTRP-GDSC) - 'CCLE_SCREENING_DATA_FOLDER' (CCLE) - 'TCGA_SCREENING_DATA' (TCGA)
		
		3 - In the 'if __name__ == "__main__":' 
			- choose the number of runs to do (e.g. k=10) 
			- choose is_test=True if you want to train on one dataset and test on another dataset or is_test=False if you want cross-validation on one dataset.



II - TriMOR-DR in 3 scripts to execute by order:

1st script: pretrain_autoencoders.py (if using MSE loss) or pretrain_autoencoders_ZINB.py (if using ZINB loss)
	Purpose: Pretrain two separate autoencoders (for cell and drug data) using all available data (without labels).
		 This step creates meaningful latent representations for the next stage.

	Steps:
		1 - In the utils.py file: Write the data modalities you want to use in this line: DATA_MODALITIES = ['cell_CN','cell_exp','cell_methy','cell_mut','drug_desc', 'drug_finger', 'drug_DT']

		2 - In the run() function, load the training data and if applicable the test data of choice by filling up this line: raw_file_directory='dataset chosen'
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)
		
		3 - In the 'if __name__ == "__main__":' choose if is_test=True if you want to pretrain with one dataset or is_test=False if you want to do the pretraining with an intersect of features between two datasets.

2nd script: train_mlp_on_latent.py or train_mlp_on_latent_LO.py for group-aware splits: Leave-Drug-Out (LDO) or Leave-Cell-Out (LCO)
	Purpose: Train an MLP on the latent representations obtained from the pretrained autoencoders, using labeled data.

	Steps:
		1 - Depending on the script executed before (pretrain_autoencoders.py or pretrain_autoencoders_ZINB.py) select the right script to import
			#from pretrain_autoencoders import SimpleAutoencoder
			#from pretrain_autoencoders_ZINB import ZINBAutoencoder

		2 - In the run() function, load the training data and if applicable the test data of choice by filling up these lines: raw_file_directory='dataset chosen'and screen_file_directory='corresponding screening file'. 
			It has to be the same dataset(s) as in the first script executed.
			Corresponding screening files for each datasets: 'BOTH_SCREENING_DATA_FOLDER' (CTRP-GDSC) - 'CCLE_SCREENING_DATA_FOLDER' (CCLE) - 'TCGA_SCREENING_DATA' (TCGA)

		3 - In the 'if __name__ == "__main__":' 
			- choose the number of runs to do (e.g. k=10) 
			- is_test=True if you want to train on one dataset and test on another dataset or is_test=False if you want cross-validation on one dataset.


3rd script: few_shot_on_TCGA.py
	Purpose: Start from the trained encoders+MLP (pick a run_id from step 2), align TCGA features using feature_columns.pkl, load train-time norms, and run a 				 sweep over K labeled TCGA samples (support) to fine-tune adapters/last layers. Produces a CSV and PNG under runs/ summarizing AUC/AUPRC vs K.

		Steps:
			1- Set run_id and the few_shot_sizes list in __main__
 
ADDITIONAL SCRIPTS USED BY AE-MLP and TriMOR-DR models:

- utils.py : Central configuration file for paths, dataset locations, and modality selection.

- data_loader.py : Loads, processes, normalizes, and intersects multi-omics data and screening matrices.

- data_loader_pretraining : Loads raw data to build full feature matrices (X_cell, X_drug) without labels, used for unsupervised autoencoder pretraining in pretrain_autoencoders.py and pretrain_autoencoders_ZINB.py

- autoencoder.py : Implements the basic autoencoder architecture used to learn low-dimensional representations of cell and drug data.

- mlp.py : Defines a simple multilayer perceptron (MLP) used for drug response classification after encoding.

- DeepDRA.py : Defines the full DeepDRA model (2 autoencoders + MLP). Includes training function with combined AE + classification loss. Returns decoded data and MLP outputs.

- evaluation.py : Provides metrics and visualizations to evaluate model performance.


III - Visualization script:

Heatmap_screening_files.py
	Purpose: Visualize the content of a drug screening matrix (values -1, 0, 1) as a heatmap.

	Steps: 
		1 - Set the file_path to the desired screening file.

		2 - Set full_matrix=True to display the full matrix or False to only show a 50x50 subset.

