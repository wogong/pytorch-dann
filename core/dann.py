"""Train dann."""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from core.test import test
from utils.utils import save_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def train_dann(model, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device):
    """Train dann."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer

    if params.src_dataset == 'mnist' or params.tgt_dataset == 'mnist':
        print("training mnist task")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        print("training office task")
        parameter_list = [{
            "params": model.features.parameters(),
            "lr": 0.001
        }, {
            "params": model.fc.parameters(),
            "lr": 0.001
        }, {
            "params": model.bottleneck.parameters()
        }, {
            "params": model.classifier.parameters()
        }, {
            "params": model.discriminator.parameters()
        }]
        optimizer = optim.SGD(parameter_list, lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # set train state for Dropout and BN layers
        model.train()
        # zip source and target data pair
        len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, class_src), (images_tgt, _)) in data_zip:

            p = float(step + epoch * len_dataloader) / \
                params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if params.src_dataset == 'mnist' or params.tgt_dataset == 'mnist':
                adjust_learning_rate(optimizer, p)
            else:
                adjust_learning_rate_office(optimizer, p)

            # prepare domain label
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            label_src = torch.zeros(size_src).long().to(device)  # source 0
            label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

            # make images variable
            class_src = class_src.to(device)
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)
            src_loss_class = criterion(src_class_output, class_src)
            src_loss_domain = criterion(src_domain_output, label_src)

            # train on target domain
            _, tgt_domain_output = model(input_data=images_tgt, alpha=alpha)
            tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

            loss = src_loss_class + src_loss_domain + tgt_loss_domain

            # optimize dann
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step == 0):
                print(
                    "Epoch [{:4d}/{}] Step [{:2d}/{}]: src_loss_class={:.6f}, src_loss_domain={:.6f}, tgt_loss_domain={:.6f}, loss={:.6f}"
                    .format(epoch + 1, params.num_epochs, step + 1, len_dataloader, src_loss_class.data.item(),
                            src_loss_domain.data.item(), tgt_loss_domain.data.item(), loss.data.item()))

        # eval model
        if ((epoch + 1) % params.eval_step == 0):
            print("eval on target domain")
            test(model, tgt_data_loader, device, flag='target')
            print("eval on source domain")
            test(model, src_data_loader, device, flag='source')

        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, params.model_root,
                       params.src_dataset + '-' + params.tgt_dataset + "-dann-{}.pt".format(epoch + 1))

    # save final model
    save_model(model, params.model_root, params.src_dataset + '-' + params.tgt_dataset + "-dann-final.pt")

    return model


def adjust_learning_rate(optimizer, p):
    lr_0 = 0.01
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_office(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr
