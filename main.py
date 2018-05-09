from models.model import CNNModel
from models.classifier import Classifier

from core.dann import train_dann
from core.test import eval, eval_src
from core.pretrain import train_src

import params
from utils import get_data_loader, init_model, init_random_seed

# init random seed
init_random_seed(params.manual_seed)

# load dataset
src_data_loader = get_data_loader(params.src_dataset)
src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

# load source classifier
src_classifier = init_model(net=Classifier(), restore=params.src_classifier_restore)

# train source model
print("=== Training classifier for source domain ===")

if not (src_classifier.restored and params.src_model_trained):
    src_classifier = train_src(src_classifier, src_data_loader)

# eval source model on both source and target domain
print("=== Evaluating source classifier for source domain ===")
eval_src(src_classifier, src_data_loader_eval)
print("=== Evaluating source classifier for target domain ===")
eval_src(src_classifier, tgt_data_loader_eval)

# load dann model
dann = init_model(net=CNNModel(), restore=params.dann_restore)

# train dann model
print("=== Training dann model ===")

if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, src_data_loader, tgt_data_loader, tgt_data_loader_eval)

# eval dann model
print("=== Evaluating dann for source domain ===")
eval(dann, src_data_loader_eval)
print("=== Evaluating dann for target domain ===")
eval(dann, tgt_data_loader_eval)

print('done')