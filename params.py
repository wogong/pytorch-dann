"""Params for DANN."""

import os

# params for path
dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN'))

# params for datasets and data loader

batch_size = 64

office_image_size = 227

# params for source dataset
src_dataset = "amazon31"
src_model_trained = True
src_classifier_restore = os.path.join(model_root,src_dataset + '-source-classifier-final.pt')
class_num_src = 31

# params for target dataset
tgt_dataset = "webcam31"
tgt_model_trained = True
dann_restore = os.path.join(model_root , src_dataset + '-' + tgt_dataset + '-dann-final.pt')

# params for pretrain
num_epochs_src = 100
log_step_src = 10
save_step_src = 20
eval_step_src = 20

# params for training dann

## for digit
# num_epochs = 400
# log_step = 100
# save_step = 20
# eval_step = 20

## for office
num_epochs = 1000
log_step = 10 # iters
save_step = 500
eval_step = 5 # epochs

manual_seed = 8888
alpha = 0

# params for optimizing models
lr = 2e-4