# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD
import os
from sklearn.metrics import roc_auc_score, roc_curve
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt', start_epoch=0):

        score = -val_loss
        if epoch >= start_epoch - 1:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
            elif score < self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience or epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

loss_fn = nn.CrossEntropyLoss()
def evaluate_val(model: ContinualModel, dataset, k, epoch, results_dir, early_stopping = None):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    # accs, accs_mask_classes = [], []
    prob_list, labels_list = [], []
    # for k, val_loader in enumerate(dataset.val_loader):
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    val_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_prob_list, all_labels_list = [], []
    eye_array = np.eye(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)
    for data in dataset.val_loader:
        with torch.no_grad():
            # inputs, labels = data
            # inputs, labels = inputs.to(model.device), labels.to(model.device)
            # import ipdb;ipdb.set_trace()
            inputs0, inputs1, labels = data
            inputs0, inputs1, labels = inputs0.to(device), inputs1.to(device), labels.to(device)
            # if 'class-il' not in model.COMPATIBILITY:
            #     outputs = model(inputs0, k)
            # else:
            outputs = model([inputs0, inputs1])

            logits, Y_prob, pred = outputs[:3]

            # _, pred = torch.max(outputs.data, 1)
            val_loss += loss_fn(logits, labels).item()
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            prob_list.append(Y_prob.cpu().numpy())
            all_prob_list.append(Y_prob.cpu().numpy())
            labels_list.append(labels.item())
            all_labels_list.append(eye_array[labels.item()])

            # if dataset.SETTING == 'class-il':
            #     # import ipdb;ipdb.set_trace()
            #     # mask_classes(outputs, dataset, k)
            #     mask_classes(logits, dataset, k)
            #     # _, pred = torch.max(outputs.data, 1)
            #     correct_mask_classes += torch.sum(pred == labels).item()
    # import ipdb;ipdb.set_trace()
    val_loss /= len(dataset.val_loader)
    auc = roc_auc_score(np.array(all_labels_list), np.concatenate(all_prob_list), multi_class='ovr')
    # acc = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
    # acc_mask_classes = correct_mask_classes / total * 100

    model.train(status)
    print(f'auc = {auc}')

    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, f"task{k}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False

class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.val_old_data = []
        # self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset, fold):
        if dataset.SETTING != 'domain-il':
            # import ipdb;ipdb.set_trace()
            # self.old_data.append(dataset.train_loader.dataset)
            # self.val_old_data.append(dataset.val_loader.dataset)
            # self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            # self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            # import ipdb;ipdb.set_trace()
            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            # for param in self.net.named_parameters():
            #     print(param[0])
            # import ipdb;ipdb.set_trace()
            # self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            self.opt = optim.Adam(self.net.parameters(), lr=self.args.lr)
            early_stopping = EarlyStopping(patience=10, stop_epoch=20, verbose=True)
            results_dir = f'./checkpoints/{self.args.exp_desc}'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # prepare dataloader
            # all_data, all_labels = None, None
            # for i in range(len(self.old_data)):
            #     if all_data is None:
            #         all_data = self.old_data[i]
            #         all_labels = self.old_labels[i]
            #     else:
            #         all_data = np.concatenate([all_data, self.old_data[i]])
            #         all_labels = np.concatenate([all_labels, self.old_labels[i]])
            # all_data0, all_data1, all_labels = [], [], []
            # for i in range(len(self.old_data)):
            #     for j in range(len(self.old_data[i])):
            #         all_data0.append(self.old_data[i][j][0])
            #         all_data1.append(self.old_data[i][j][1])
            #         all_labels.append(self.old_data[i][j][0])
            
            # val_all_data0, val_all_data1, val_all_labels = [], [], []
            # for i in range(len(self.val_old_data)):
            #     for j in range(len(self.val_old_data[i])):
            #         val_all_data0.append(self.val_old_data[i][j][0])
            #         val_all_data1.append(self.val_old_data[i][j][1])
            #         val_all_labels.append(self.val_old_data[i][j][0])
            
            # import ipdb;ipdb.set_trace()

            # # transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            # train_dataset = ValidationDataset(all_data0, all_data1, all_labels)
            # loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            # val_dataset = ValidationDataset(val_all_data0, val_all_data1, val_all_labels)
            # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

            train_loader = dataset.train_loader
            val_loader = dataset.val_loader
            # results_dir = './results/joint'
            # train
            for e in range(self.args.n_epochs):
                # stop = evaluate_val(self.net, dataset, dataset.N_TASKS-1, epoch=e, results_dir=results_dir, early_stopping=early_stopping)
                barlow_loss = 0
                for i, batch in enumerate(train_loader):
                    # import ipdb;ipdb.set_trace()
                    inputs0, inputs1, labels = batch
                    inputs0, inputs1, labels = inputs0.to(self.device), inputs1.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net([inputs0, inputs1])
                    # import ipdb;ipdb.set_trace()
                    # loss = self.loss(outputs[0], labels.long()) + 0.00001 * outputs[-1].mean()
                    # barlow_loss += outputs[-1]
                    # loss = self.loss(outputs[0], labels.long()) + 0.001 * outputs[-1]
                    loss = self.loss(outputs[0], labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(train_loader), e, 'J', loss.item(), fold)
                print('\n')
                # import ipdb;ipdb.set_trace()
                # print(f'Barlow-Twins Loss: {barlow_loss.item()}')
                stop = evaluate_val(self.net, dataset, dataset.N_TASKS-1, epoch=e, results_dir=results_dir, early_stopping=early_stopping)
                if stop:
                    break
            self.net.load_state_dict(torch.load(os.path.join(results_dir, f"task{dataset.N_TASKS-1}_checkpoint.pt")))
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            
            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i+1) * bs]
                    labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())
                
                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
