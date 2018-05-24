import torch.utils.data
import torch.nn as nn

from utils import make_variable

def test_from_save(model, saved_model, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    classifier = model.load_state_dict(torch.load(saved_model))
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.NLLLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels) #labels = labels.squeeze(1)
        preds = classifier(images)

        criterion(preds, labels)

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:.2%}".format(loss, acc))

def eval(model, data_loader):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0
    acc_domain = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels) #labels = labels.squeeze(1)
        size_tgt = len(labels)
        labels_domain = make_variable(torch.ones(size_tgt).long())

        preds, domain = model(images, alpha=0)

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()
        acc_domain += pred_domain.eq(labels_domain.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    acc_domain /= len(data_loader.dataset)

    print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}, Avg Domain Accuracy = {:2%}".format(loss, acc, acc_domain))

def eval_src(model, data_loader):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0
    acc_domain = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels) #labels = labels.squeeze(1)
        size_tgt = len(labels)
        labels_domain = make_variable(torch.zeros(size_tgt).long())

        preds, domain = model(images, alpha=0)

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()
        acc_domain += pred_domain.eq(labels_domain.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    acc_domain /= len(data_loader.dataset)

    print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}, Avg Domain Accuracy = {:2%}".format(loss, acc, acc_domain))

def eval_src_(model, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.NLLLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels) #labels = labels.squeeze(1)
        preds = model(images)

        criterion(preds, labels)

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()


    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}".format(loss, acc))