import os
import sys
import datetime
from tensorboardX import SummaryWriter

import torch
sys.path.append('../')
from models.model import GTSRBmodel
from core.dann import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for path
    dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
    model_name = "synsigns-gtsrb"
    model_base = '/home/wogong/models/pytorch-dann'
    note = ''
    now = datetime.datetime.now().strftime('%m%d_%H%M%S')
    model_root = os.path.join(model_base, model_name, note + '_' + now)
    finetune_flag = False

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "synsigns"
    source_image_root = os.path.join('/home/wogong/datasets', 'synsigns')
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    tgt_dataset = "gtsrb"
    target_image_root = os.path.join('/home/wogong/datasets', 'gtsrb')
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 20

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 200
    log_step = 50
    save_step = 100
    eval_step = 5

    manual_seed = None
    alpha = 0

    # params for optimizing models
    lr = 2e-4

params = Config()
logger = SummaryWriter(params.model_root)
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.source_image_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.source_image_root, params.batch_size, train=False)
tgt_data_loader, tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.target_image_root, params.batch_size, train=True)

# load dann model
dann = init_model(net=GTSRBmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger)
