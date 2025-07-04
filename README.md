Description des scripts et ordre d'exécution – Pipeline DeepDRA and AEMOCDR

Original DeepDRA script: main.py and main_clinical.py:

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



AEMOCDR (my model) in 3 scripts to execute by order:

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
 
