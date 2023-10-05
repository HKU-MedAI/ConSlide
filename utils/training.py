# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torch.nn.functional as F

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

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt', start_epoch=6):

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

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    # prob_list, labels_list = [], []
    accs, accs_mask_classes = [], []
    aucs, aucs_mask_classes = [], []
    all_prob_list, all_labels_list = [], []
    eye_array = np.eye(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)
    num_task = len(dataset.test_loaders)

    corrects = 0
    corrects_mask_classes = 0
    totals = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        prob_list, mask_prob_list, labels_list = [], [], []
        for data in test_loader:
            
            with torch.no_grad():
                # inputs, labels = data
                # inputs, labels = inputs.to(model.device), labels.to(model.device)
                inputs0, inputs1, labels = data
                inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs0, k)
                else:
                    # import ipdb;ipdb.set_trace()
                    outputs = model([inputs0, inputs1])

                logits, Y_prob, pred = outputs[:3]

                # _, pred = torch.max(outputs.data, 1)
                # import ipdb;ipdb.set_trace()
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                prob_list.append(Y_prob.cpu().numpy())
                all_prob_list.append(Y_prob.cpu().numpy()[0][:2*num_task])
                labels_list.append(labels.item())
                all_labels_list.append(eye_array[labels.item()][:2*num_task])

                if dataset.SETTING == 'class-il':
                    # import ipdb;ipdb.set_trace()
                    # mask_classes(outputs, dataset, k)
                    mask_classes(logits, dataset, k)
                    _, pred = torch.max(logits, 1)
                    mask_prob = F.softmax(logits, dim = 1)
                    mask_prob_list.append(mask_prob.cpu().numpy())
                    correct_mask_classes += torch.sum(pred == labels).item()
        # import ipdb;ipdb.set_trace()
        aucs.append(roc_auc_score(np.array(labels_list), np.concatenate(prob_list)[:, 2*k + 1]))
        aucs_mask_classes.append(roc_auc_score(np.array(labels_list) - (2*k), np.concatenate(mask_prob_list)[:, 2*k + 1]))
        accs.append(correct / total if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total)

        corrects += correct
        corrects_mask_classes += correct_mask_classes
        totals += total
    # import ipdb;ipdb.set_trace()
    micro_acc = corrects / totals
    micro_acc_mask_classes = corrects_mask_classes / totals

    model.net.train(status)
    # import ipdb;ipdb.set_trace()
    if not last:
        all_aucs = roc_auc_score(np.array(all_labels_list), np.array(all_prob_list), multi_class='ovr')
        return [accs, micro_acc, accs_mask_classes, micro_acc_mask_classes, aucs, aucs_mask_classes, all_aucs]
    else:
        return [accs, micro_acc, accs_mask_classes, micro_acc_mask_classes, aucs, aucs_mask_classes]

loss_fn = nn.CrossEntropyLoss()
def evaluate_val(model: ContinualModel, dataset: ContinualDataset, k, epoch, results_dir, early_stopping = None):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    # accs, accs_mask_classes = [], []
    prob_list, labels_list = [], []
    # for k, val_loader in enumerate(dataset.val_loader):
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    val_loss = 0
    for data in dataset.val_loader:
        with torch.no_grad():
            # inputs, labels = data
            # inputs, labels = inputs.to(model.device), labels.to(model.device)
            # import ipdb;ipdb.set_trace()
            inputs0, inputs1, labels = data
            inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs0, k)
            else:
                outputs = model([inputs0, inputs1])

            logits, Y_prob, pred = outputs[:3]

            # _, pred = torch.max(outputs.data, 1)
            val_loss += loss_fn(logits, labels).item()
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            prob_list.append(Y_prob.cpu().numpy())
            labels_list.append(labels.item())

            if dataset.SETTING == 'class-il':
                # import ipdb;ipdb.set_trace()
                # mask_classes(outputs, dataset, k)
                mask_classes(logits, dataset, k)
                # _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
    # import ipdb;ipdb.set_trace()
    val_loss /= len(dataset.val_loader)
    auc = roc_auc_score(np.array(labels_list), np.concatenate(prob_list)[:, 2*k + 1])
    acc = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
    acc_mask_classes = correct_mask_classes / total * 100

    model.net.train(status)
    print(f'\t auc = {auc}')

    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, f"task{k}_checkpoint.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False
    # return [acc, acc_mask_classes, auc]


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace, fold) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    acc_results, micro_acc_results, acc_results_mask_classes, micro_acc_results_mask_classes = [], [], [], []
    auc_results = []
    # early_stopping = EarlyStopping(patience = 20, stop_epoch=30, verbose = True)

    # import ipdb;ipdb.set_trace()
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, fold, args.exp_desc)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _, _ = dataset_copy.get_data_loaders(fold)
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        acc_random_results_class, micro_acc_random_results_class, acc_random_results_task, micro_acc_random_results_task, auc_random_results_class, _, all_auc_random_results_class = evaluate(model, dataset_copy)
        print(f'Random AUC = {all_auc_random_results_class}')
    print(file=sys.stderr)
    if model.NAME != 'joint':
        for t in range(dataset.N_TASKS):
            early_stopping = EarlyStopping(patience=10, stop_epoch=10, verbose=True)
            # results_dir = f'./check_points/{model.NAME}'
            results_dir = f'./checkpoints/{args.exp_desc}'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            model.net.train()
            train_loader, val_loader, test_loader = dataset.get_data_loaders(fold)
            if hasattr(model, 'begin_task'):
                # import ipdb;ipdb.set_trace()
                model.begin_task(dataset)
            if t:
                accs = evaluate(model, dataset, last=True)
                # import ipdb;ipdb.set_trace()
                acc_results[t-1] = acc_results[t-1] + accs[0]
                auc_results[t-1] = auc_results[t-1] + accs[2]
                if dataset.SETTING == 'class-il':
                    acc_results_mask_classes[t-1] = acc_results_mask_classes[t-1] + accs[2]
                    # auc_results_mask_classes[t-1] = auc_results_mask_classes[t-1] + accs[3]

            scheduler = dataset.get_scheduler(model, args)
            # import ipdb;ipdb.set_trace()
            # val_accs = evaluate_val(model, dataset, t)
            # import ipdb;ipdb.set_trace()

            # accs = evaluate(model, dataset)
            if 1:
                if t == 0:
                    for epoch in range(10):
                        loss = 0
                        for i, data in enumerate(train_loader):
                            inputs0, inputs1, labels = data
                            inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                            loss += model.observe(inputs0, inputs1, labels, t, ssl=True)

                            progress_bar(i, len(train_loader), epoch, t, loss, fold)
                        print(f'SSL Loss: {loss}')

            for epoch in range(model.args.n_epochs):
                for i, data in enumerate(train_loader):
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs0, inputs1, labels, logits = data
                        inputs0, inputs1, labels, logits = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device), logits.to(model.device)
                        loss = model.observe(inputs0, inputs1, labels, logits)
                    else:
                        inputs0, inputs1, labels = data
                        inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                        loss = model.observe(inputs0, inputs1, labels, t, ssl=False)

                    progress_bar(i, len(train_loader), epoch, t, loss, fold)

                    if args.tensorboard:
                        tb_logger.log_loss(loss, args, epoch, t, i)
                # stop = evaluate_val(model, dataset, t, epoch=epoch, results_dir=results_dir, early_stopping=early_stopping)
                stop = evaluate_val(model, dataset, t, epoch=epoch, results_dir=results_dir, early_stopping=early_stopping)
                if stop:
                    break
                if scheduler is not None:
                    scheduler.step()
            model.load_state_dict(torch.load(os.path.join(results_dir, f"task{t}_checkpoint.pt")))

            # Add buffer data
            if hasattr(model, 'save_buffer'):
                # for epoch in range(model.args.n_epochs):
                for i, data in enumerate(train_loader):
                    inputs0, inputs1, labels = data
                    inputs0, inputs1, labels = inputs0.to(model.device), inputs1.to(model.device), labels.to(model.device)
                    model.save_buffer(inputs0, inputs1, labels, t)

            if hasattr(model, 'end_task'):
                model.end_task(dataset)

            accs = evaluate(model, dataset)
            # import ipdb;ipdb.set_trace()
            acc_results.append(accs[0])
            micro_acc_results.append(accs[1])
            auc_results.append(accs[5])
            acc_results_mask_classes.append(accs[2])
            micro_acc_results_mask_classes.append(accs[3])
            # auc_results_mask_classes.append(accs[3])
            print('\n')
            print(f'acc:                {accs[0]}')
            print(f'macro acc:          {np.mean(accs[0])}')
            print(f'micro acc:          {accs[1]}')
            print(f'mask acc:           {accs[2]}')
            print(f'macro mask acc:     {np.mean(accs[2])}')
            print(f'micro mask acc:     {accs[3]}')
            print(f'auc:                {accs[5]}')
            print(f'multi-classes auc:  {accs[6]}')
            print('\n')

            # import ipdb;ipdb.set_trace()
            mean_acc = np.mean(accs[0])
            micro_acc = accs[1]
            mean_acc_mask = np.mean(accs[2])
            micro_acc_mask = accs[3]
            mean_auc = np.mean(accs[5])
            multi_class_auc = accs[6]
            # import ipdb;ipdb.set_trace()
            # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            if args.csv_log:
                csv_logger.log(mean_acc, mean_acc_mask, micro_acc, micro_acc_mask, mean_auc, multi_class_auc)
            if args.tensorboard:
                tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
    else:
        _, _, _ = dataset.get_joint_data_loaders(fold)
        if hasattr(model, 'end_task'):
            model.end_task(dataset, fold)
        accs = evaluate(model, dataset)
            # import ipdb;ipdb.set_trace()
        acc_results.append(accs[0])
        micro_acc_results.append(accs[1])
        auc_results.append(accs[5])
        acc_results_mask_classes.append(accs[2])
        micro_acc_results_mask_classes.append(accs[3])
        # auc_results_mask_classes.append(accs[3])
        print('\n')
        print(f'acc:                {accs[0]}')
        print(f'macro acc:          {np.mean(accs[0])}')
        print(f'micro acc:          {accs[1]}')
        print(f'mask acc:           {accs[2]}')
        print(f'macro mask acc:     {np.mean(accs[2])}')
        print(f'micro mask acc:     {accs[3]}')
        print(f'auc:                {accs[5]}')
        print(f'multi-classes auc:  {accs[6]}')
        print('\n')

        # import ipdb;ipdb.set_trace()
        mean_acc = np.mean(accs[0])
        micro_acc = accs[1]
        mean_acc_mask = np.mean(accs[2])
        micro_acc_mask = accs[3]
        mean_auc = np.mean(accs[5])
        multi_class_auc = accs[6]
        # import ipdb;ipdb.set_trace()
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            csv_logger.log(mean_acc, mean_acc_mask, micro_acc, micro_acc_mask, mean_auc, multi_class_auc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        # import ipdb;ipdb.set_trace()
        csv_logger.add_bwt(acc_results, acc_results_mask_classes, auc_results)
        csv_logger.add_forgetting(acc_results, acc_results_mask_classes, auc_results)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(acc_results, acc_random_results_class,
                               acc_results_mask_classes, acc_random_results_task,
                               auc_results, auc_random_results_class)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
