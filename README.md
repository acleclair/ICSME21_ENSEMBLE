# ICSME21_ENSEMBLE
Reproducibility Package for ICSME21 Paper on Model Ensembles for Source Code Summarization

Download Attn-to-Fc repo from https://github.com/Attn-to-FC/Attn-to-FC

Download the train_ensemble.py and predict_ensemble.py files from this repository.

Example to train a model using the Bagging ensemble technique:

python3 train.py --model-type=ast-attendgru-fc --gpu=0 --bagging

To predict an ensemble of two models use the command:

python predict_ensemble.py  {path_to_model_1} {path_to_model_2} --gpu 1
