"""Train dann."""

import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
import numpy as np
from core.test import eval

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def train_dann(dann, src_data_loader, tgt_data_loader, tgt_data_loader_eval):
    """Train dann."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    dann.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(dann.parameters(), lr=params.lr)

    criterion = nn.NLLLoss()

    for p in dann.parameters():
        p.requires_grad = True

    ####################
    # 2. train network #
    ####################

    # prepare domain label
    label_src = make_variable(torch.zeros(params.batch_size).long())  # source 0
    label_tgt = make_variable(torch.ones(params.batch_size).long())  # target 1

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, class_src), (images_tgt, _)) in data_zip:

            p = float(step + epoch * len_dataloader) / params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # make images variable
            class_src = make_variable(class_src)
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = dann(input_data=images_src, alpha=alpha)
            src_loss_class = criterion(src_class_output, class_src)
            src_loss_domain = criterion(src_domain_output, label_src)

            # train on target domain
            _, tgt_domain_output = dann(input_data=images_tgt, alpha=alpha)
            tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

            loss = src_loss_class + src_loss_domain + tgt_loss_domain

            # optimize dann
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: src_loss_class={}, src_loss_domain={}, tgt_loss_domain={}, loss={}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_dataloader,
                              src_loss_class.data[0],
                              src_loss_domain.data[0],
                              tgt_loss_domain.data[0],
                              loss.data[0]))

        # eval model on test set
        if ((epoch + 1) % params.eval_step == 0):
            eval(dann, tgt_data_loader_eval)
            dann.train()

        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(dann, params.src_dataset + '-' + params.tgt_dataset + "-dann-{}.pt".format(epoch + 1))

    # save final model
    save_model(dann, params.src_dataset + '-' + params.tgt_dataset + "-dann-final.pt")

    return dann