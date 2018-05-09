import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import torch.nn as nn

import params
from utils import make_variable

def test(dataloader, epoch):

    if torch.cuda.is_available():
        cuda = True
        cudnn.benchmark = True

    """ training """

    my_net = torch.load(os.path.join(
        params.model_root, params.src_dataset + '_' + params.tgt_dataset + '_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, params.image_size, params.image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        class_output, _ = my_net(input_data=inputv_img, alpha=params.alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct * 1.0 / n_total

    print 'epoch: %d, accuracy: %f' % (epoch, accu)


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

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def eval(model, data_loader):
    """Evaluate model for dataset."""
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
        preds, _ = model(images, alpha=0)

        criterion(preds, labels)

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def eval_src(model, data_loader):
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

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))