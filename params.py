"""Params for DANN."""

import os

# params for path
dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN'))

# params for datasets and data loader
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 128
image_size = 28

# params for source dataset
src_dataset = "SVHN"
src_model_trained = True
src_classifier_restore = os.path.join(model_root,src_dataset + '-source-classifier-final.pt')

# params for target dataset
tgt_dataset = "MNIST"
tgt_model_trained = True
dann_restore = os.path.join(model_root , src_dataset + '-' + tgt_dataset + '-dann-final.pt')

# params for pretrain
num_epochs_src = 100
log_step_src = 10
save_step_src = 20
eval_step_src = 20

# params for training dann
num_epochs = 400
log_step = 50
save_step = 50
eval_step = 20

manual_seed = 8888
alpha = 0

# params for optimizing models
lr = 2e-4