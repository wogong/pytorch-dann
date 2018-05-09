"""Train classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
from core.test import eval_src

def train_src(model, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    model.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_class = nn.NLLLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_src):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = model(images)
            loss = loss_class(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_src == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_src,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_src == 0):
            eval_src(model, data_loader)
            model.train()

        # save model parameters
        if ((epoch + 1) % params.save_step_src == 0):
            save_model(model, params.src_dataset + "-source-classifier-{}.pt".format(epoch + 1))

    # save final model
    save_model(model, params.src_dataset + "-source-classifier-final.pt")

    return model