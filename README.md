Description of Scripts and Execution Order – DeepDRA and AEMOCDR Pipeline

1 - Original DeepDRA script: main.py and main_clinical.py:

	Purpose: These scripts execute the full training and evaluation pipeline using the original version of DeepDRA (with or without clinical data). 
		 Beware that only CCLE dataset has a clinical data file !

	Steps:
		1 - In the utils.py file: Write the data modalities you want to use in this line: DATA_MODALITIES = ['cell_CN','cell_exp','cell_methy','cell_mut','drug_desc', 'drug_finger', 'drug_DT']

		2 - In the run() function, load the training data and if applicable the test data of choice by filling up these lines: raw_file_directory='dataset chosen'and screen_file_directory='corresponding screening file'
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)
			Corresponding screening files for each datasets: 'BOTH_SCREENING_DATA_FOLDER' (CTRP-GDSC) - 'CCLE_SCREENING_DATA_FOLDER' (CCLE) - 'TCGA_SCREENING_DATA' (TCGA)
		
		3 - In the 'if __name__ == "__main__":' 
			- choose the number of runs to do (e.g. k=10) 
			- choose is_test=True if you want to train on one dataset and test on another dataset or is_test=False if you want cross-validation on one dataset.



2 - AEMOCDR (my model) in 3 scripts to execute by order:

1st script: pretrain_autoencoders.py (if using MSE loss) or pretrain_autoencoders_ZINB.py (if using ZINB loss)
	Purpose: Pretrain two separate autoencoders (for cell and drug data) using all available data (without labels).
		 This step creates meaningful latent representations for the next stage.

	Steps:
		1 - In the utils.py file: Write the data modalities you want to use in this line: DATA_MODALITIES = ['cell_CN','cell_exp','cell_methy','cell_mut','drug_desc', 'drug_finger', 'drug_DT']

		2 - In the run() function, load the training data and if applicable the test data of choice by filling up this line: raw_file_directory='dataset chosen'
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)
		
		3 - In the 'if __name__ == "__main__":' choose if is_test=True if you want to pretrain with one dataset or is_test=False if you want to do the pretraining with an intersect of features between two datasets.

2nd script: train_mlp_on_latent.py
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


3rd script: fine_tune_TCGA.py or Test_on_TCGA.py
	Purpose: Fine-tune the model (encoders + MLP) on a subset of TCGA samples to improve prediction on this target dataset (fine_tune_TCGA.py) and test on the remaining TCGA samples not used for training.
		 Evaluate the final model on the entire TCGA dataset without fine-tuning (Test_on_TCGA.py).

	Steps:
		1 - In this line : train_len = int(0.0145 * len(dataset)), Choose the percentage of TCGA samples to be used for fine-tuning

		2 - In the 'if __name__ == "__main__":' 
			- choose the run_id number from 2nd script that you want to use to load the choosen model
 
ADDITIONAL SCRIPTS USED BY DeepDRA and AEMOCDR models:

- utils.py : Central configuration file for paths, dataset locations, and modality selection.

- data_loader.py : Loads, processes, normalizes, and intersects multi-omics data and screening matrices.

- data_loader_clinical.py : Same role as data_loader.py but adds a load_clinical_data() method to one-hot encode and normalize clinical data. Used only by the main_clinical.py script

- data_loader_pretraining : Loads raw data to build full feature matrices (X_cell, X_drug) without labels, used for unsupervised autoencoder pretraining in pretrain_autoencoders.py and pretrain_autoencoders_ZINB.py

- autoencoder.py : Implements the basic autoencoder architecture used to learn low-dimensional representations of cell and drug data.

- mlp.py : Defines a simple multilayer perceptron (MLP) used for drug response classification after encoding.

- DeepDRA.py : Defines the full DeepDRA model (2 autoencoders + MLP). Includes training function with combined AE + classification loss. Returns decoded data and MLP outputs.

- DeepDRA_clinical.py : Extension of DeepDRA to include a clinical branch. Takes clinical data as additional input to the MLP. Used only by the main_clinical.py script

- evaluation.py : Provides metrics and visualizations to evaluate model performance.



3 - Data exploration scripts:

3.1 - Script: resistant_sensitive_screening_file_counts.py
	Purpose: Count the number of resistant (1), sensitive (-1), and unknown (0) values in the drug screening files of the CTRP-GDSC and CCLE datasets.

	Steps: Run the script directly (no arguments required). It will print the counts for each value type in both datasets.

3.2 - Script: raw_data_shape.ipynb
	Purpose: This script is used to explore the structure and content of raw data files from all datasets (CTRP, GDSC, CCLE, TCGA). 
		 It prints the shape (samples × features) of each modality (expression, mutation, copy number, methylation, etc.) and provides statistics like total mutation counts or feature summaries. 
		 Useful for verifying data availability and format consistency before preprocessing or training models.


4 - Visualization Scripts:

4.1 - Script: t-SNE_new_DeepDRA.py
	Purpose: Visualize cell and drug data before and after encoding using pretrained autoencoders with t-SNE.

	Steps: 
		1 - Ensure encoder_cell.pth and encoder_drug.pth are available in the working directory. They are generated by pretrain_autoencoders.py or pretrain_autoencoders_ZINB.py

		2 - Check and set DATA_MODALITIES in utils.py.

		3 - Choose the raw_file_directory of the dataset you want to visualize
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)

4.2 - Scripts: cell_t-sne.py and Drug_t-sne.py
	Purpose: Visualize with a t-SNE how the cell data or drug data is distributed in raw and latent spaces, using the original DeepDRA autoencoder trained on resistance/sensitivity labels.
		 The script prepares training data, trains an autoencoder, computes latent representations, and plots t-SNE for raw and latent features. 
		 Labels are inferred from the screening file to color points as resistant/sensitive.

	Steps:

		1 - Set data_modalities. For each script you have to put at least one modality of the other type of data as we need the labels. 
			For example if you want to visualize 'cell_exp' using cell_t-sne.py script. You also have to put a drug modality like 'drug_finger' for example.

		2 - Choose the raw_file_directory and screen_file_directory of the dataset you want to visualize
			datasets available: 'RAW_BOTH_DATA_FOLDER' (CTRP-GDSC: cell dataset1) - 'CCLE_RAW_DATA_FOLDER' (CCLE: cell dataset2) - 'TCGA_DATA_FOLDER' (TCGA: patient dataset)
			Corresponding screening files for each datasets: 'BOTH_SCREENING_DATA_FOLDER' (CTRP-GDSC) - 'CCLE_SCREENING_DATA_FOLDER' (CCLE) - 'TCGA_SCREENING_DATA' (TCGA)

4.3 - Script: Heatmap_screening_files.py
	Purpose: Visualize the content of a drug screening matrix (values -1, 0, 1) as a heatmap.

	Steps: 
		1 - Set the file_path to the desired screening file.

		2 - Set full_matrix=True to display the full matrix or False to only show a 50x50 subset.

4.4 - Script: Performance_vs_nbr_TCGA_sample_fine_tuning.py
	Purpose: Plot how performance metrics (Accuracy, Precision, Recall, F1 score, AUC, AUPRC) vary depending on the number of TCGA samples used for fine-tuning.
	
	Steps: 
		1 - Write the data you have for each metrics. 
		
		2 - Choose the metrics you want to plot in 'metrics =". It will plot a line chart showing performance evolution for different fine-tuning sample sizes.

5 - Mutation modality from TCGA patients Scripts:

5.1 - Script: extract_patient_id_of_cell_exp_tcga_file.py
	Purpose: Extract and save the list of patient IDs from the TCGA expression data file.
		 It reads cell_exp_raw.gzip and saves all patient IDs to TCGA_cell_exp_sample_ids.txt
		 This file has been uploaded on the GDC Data Portal to obtain other type of data modalities (mutations / clinical / CNV) for TCGA patients.

5.2 - Script: DeepDRA_TCGA_mutation.ipynb
	Purpose: From the file metadata.cohort.2025-06-12.json downloaded on the GDC Data Portal, it extracts entity_submitter_id and case_id pairs from the JSON structure.
		 It then filters the extracted IDs to retain only those that are also present in the cell_exp TCGA file (thanks to the TCGA_IDs file generated by tge raw_data_shape.ipynb script). 
		 The output is saved in a tcga_ids_mapping.tsv file.

5.3 - Script: mutation_tcga_file.ipynb
	Purpose: This script processes all TCGA MAF (Mutation Annotation Format) files from the GDC repository (downloaded on the GDC Data Portal). 
		 It merges them, filters by the gene list of TCGA_mutation_gene_list.csv generated by the raw_data_shape.ipynb script.
		 It uses the mapping file tcga_ids_mapping.tsv generated by script DeepDRA_TCGA_mutation.ipynb to retain only data from patients that are present in the cell_exp file of TCGA data.
		 It removes irrelevant or silent mutations, and generates a binary mutation matrix (patients × genes) for downstream use in DeepDRA or AEMOCDR as cell_mut.

	

6 - Clinical modality from CCLE dataset Script: CCLE_clinical_data_file_creation.ipynb
	Purpose: This script identifies cell lines that are shared across all CCLE data files (expression, mutation, CNV, screening) and saves the result in a file: unique_cell_lines_per_file.tsv
		 The cell lines are then cross-referenced with CCLE clinical annotations from the cell_lines_annotation_20181226.txt file downloaded on the CCLE database. 
		 It generates a filtered clinical file (matched_cell_lines_annotations.tsv) that contains relevant phenotypic information (e.g., pathology, age, gender), ready to be used in the DeepDRA pipeline with clinical inputs.
