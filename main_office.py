from models.model import SVHNmodel, Classifier

from core.dann import train_dann
from core.test import eval
from models.model import AlexModel

import params
from utils import get_data_loader, init_model, init_random_seed

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