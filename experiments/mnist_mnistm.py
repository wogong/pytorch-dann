import os
import sys

sys.path.append('../')
from models.model import MNISTmodel
from core.dann import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for path
    dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN'))

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "mnist"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
    class_num_src = 31

    # params for target dataset
    tgt_dataset = "mnistm"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 20

    # params for training dann

    ## for digit
    num_epochs = 100
    log_step = 20
    save_step = 50
    eval_step = 5

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs

    manual_seed = 8888
    alpha = 0

    # params for optimizing models
    lr = 2e-4

params = Config()

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=True)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=False)

# load dann model
dann = init_model(net=MNISTmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval)

# eval dann model
print("Evaluating dann for source domain {}".format(params.src_dataset))
eval(dann, src_data_loader_eval)
print("Evaluating dann for target domain {}".format(params.tgt_dataset))
eval(dann, tgt_data_loader_eval)