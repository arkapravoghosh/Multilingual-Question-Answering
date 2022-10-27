# Multilingual-Question-Answering

# About
This document is a guide for the code base. It contains description of source code files, commands for running the scripts and installation instructions.

## Scripts

1. IndicQuestionAnswering.py
This script contains code for converting AI4Bharat dataset to SQuAD format. Convertion to SQuAD format is necessary since the main dataset TyDi-QA is also in SQuAD format. 

2. run_qa.py
Tis script contains code to train model on the entire TyDIQA dataset. 

3. evaluate_qa.py
This script contains code to evaulate a trained model (downloaded from HF) on the TyDI-QA dataset

4. subset_qa.py
This script contains code to find subsets of data from a source dataset which have atleast 60% F1-score. The subset dataset obtained is augmented with TyDi-QA data and finally uploaded to HuggingFace (HF) so that in later scripts it can be fetched from HF

5. train_qa.py
This script contains code to run training and validation on the augmented_data which is downloaded from HuggingFace. It also writes the F1-scores for each data instance in the TyDI-QA validatoin set to the file "evaluation.csv"

6. trainer_qa.py
This is a helper file 

7. utils_qa.py
This is a helper file 

## Workflow

Steps:
1. Run run_qa.py to train the model and view the validation performance. 
2. Run evaluate_qa.py to obtain the view the valiation performance only
3. Run subset_qa.pt to obtain the subsets from AI4Bharat data and upload this to HuggingFace
4. Run train_qa.py to run training on the augmented data and view validation performance.

## Commands for running scripts

1. Command to train mBERT model for 3 epochs

python run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name tydiqa \
  --dataset_config_name secondary_task \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir train_epoch_3


2. Command to evaluate our model on Validation Data

python evaluate_qa.py \
  --model_name_or_path <model trained in step 1> \
  --dataset_name tydiqa \
  --dataset_config_name secondary_task \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir evaluate


3. Command for predicting on Squad, AIBharat Bengali and Telugu dataset and choosing a subset for self-training

python subset_qa.py \
  --model_name_or_path <model trained in step 1> \
  --dataset_name augment_data \
  --do_predict \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 6000 \
  --overwrite_output_dir \
  --output_dir subset

4. Command to append the selected data and train the model on the new dataset and evaluate on the dev set

python train_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name horsbug98/squad_ai4bharat_ben_tel_train \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \`
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 6000 \
  --output_dir augment_train_3
