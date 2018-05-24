import os
from core.dann import train_dann
from core.test import eval
from models.model import AlexModel

from utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for path
    dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN'))

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "amazon31"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    tgt_dataset = "webcam31"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 20

    # params for training dann

    ## for office
    num_epochs = 1000
    log_step = 10  # iters
    save_step = 500
    eval_step = 5  # epochs

    manual_seed = 8888
    alpha = 0

    # params for optimizing models
    lr = 2e-4

params = Config()

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_data_loader(params.src_dataset)
tgt_data_loader = get_data_loader(params.tgt_dataset)

# load dann model
dann = init_model(net=AlexModel(), restore=params.dann_restore)

# train dann model
print("Start training dann model.")

if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, src_data_loader, tgt_data_loader, tgt_data_loader)

# eval dann model
print("Evaluating dann for source domain")
eval(dann, src_data_loader)
print("Evaluating dann for target domain")
eval(dann, tgt_data_loader)

print('done')