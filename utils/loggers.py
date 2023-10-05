# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, fold, exp_desc) -> None:
        self.fold = fold
        self.exp_desc = exp_desc
        self.multi_class_auc = []
        self.accs = []
        self.micro_accs = []
        self.aucs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.micro_accs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.acc_fwt = None
        self.micro_acc_fwt = None
        self.auc_fwt = None
        self.acc_fwt_mask_classes = None
        self.micro_acc_fwt_mask_classes = None
        # self.auc_fwt_mask_classes = None
        self.acc_bwt = None
        self.auc_bwt = None
        self.acc_bwt_mask_classes = None
        # self.auc_bwt_mask_classes = None
        self.acc_forgetting = None
        self.auc_forgetting = None
        self.acc_forgetting_mask_classes = None
        # self.auc_forgetting_mask_classes = None

    def add_fwt(self, acc_results, acc_accs, acc_results_mask_classes, acc_accs_mask_classes, auc_results, auc_accs):
        self.acc_fwt = forward_transfer(acc_results, acc_accs)
        self.auc_fwt = forward_transfer(auc_results, auc_accs)
        if self.setting == 'class-il':
            self.acc_fwt_mask_classes = forward_transfer(acc_results_mask_classes, acc_accs_mask_classes)
            # self.auc_fwt_mask_classes = forward_transfer(auc_results_mask_classes, auc_accs_mask_classes)

    def add_bwt(self, acc_results, acc_results_mask_classes, auc_results):
        # import ipdb;ipdb.set_trace()
        self.acc_results_mask_classes = acc_results_mask_classes
        # self.auc_results_mask_classes = auc_results_mask_classes
        self.acc_bwt = backward_transfer(acc_results)
        self.acc_bwt_mask_classes = backward_transfer(acc_results_mask_classes)
        self.auc_bwt = backward_transfer(auc_results)
        # self.auc_bwt_mask_classes = backward_transfer(auc_results_mask_classes)

    def add_forgetting(self, acc_results, acc_results_mask_classes, auc_results):
        self.acc_forgetting = forgetting(acc_results)
        self.acc_forgetting_mask_classes = forgetting(acc_results_mask_classes)
        self.auc_forgetting = forgetting(auc_results)
        # self.auc_forgetting_mask_classes = forgetting(auc_results_mask_classes)

    def log(self, mean_acc, mean_acc_mask, micro_acc, micro_acc_mask, mean_auc, multi_class_auc) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        # if self.setting == 'general-continual':
        #     self.accs.append(mean_acc)
        # elif self.setting == 'domain-il':
        #     mean_acc, _ = mean_acc
        #     self.accs.append(mean_acc)
        # else:
        # mean_acc_class_il, mean_acc_task_il = mean_acc
        self.accs.append(mean_acc)
        self.micro_accs.append(micro_acc)
        self.accs_mask_classes.append(mean_acc_mask)
        self.micro_accs_mask_classes.append(micro_acc_mask)
        self.aucs.append(mean_auc)
        self.multi_class_auc.append(multi_class_auc)
        # self.aucs_mask_classes.append(mean_auc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []

        args['task_macro_acc_last'] = format(self.accs[-1], '.4f')
        new_cols.append('task_macro_acc_last')
        args['task_micro_acc_last'] = format(self.micro_accs[-1], '.4f')
        new_cols.append('task_micro_acc_last')
        args['task_macro_acc_mask_last'] = format(self.accs_mask_classes[-1], '.4f')
        new_cols.append('task_macro_acc_mask_last')
        args['task_micro_acc_mask_last'] = format(self.micro_accs_mask_classes[-1], '.4f')
        new_cols.append('task_micro_acc_mask_last')
        args['auc_last'] = format(self.aucs[-1], '.4f')
        new_cols.append('auc_last')
        args['multi_class_auc_last'] = format(self.multi_class_auc[-1], '.4f')
        new_cols.append('multi_class_auc_last')

        args['task_macro_acc_mean'] = format(np.mean(self.accs), '.4f')
        new_cols.append('task_macro_acc_mean')
        args['task_micro_acc_mean'] = format(np.mean(self.micro_accs), '.4f')
        new_cols.append('task_micro_acc_mean')
        args['task_macro_acc_mask_mean'] = format(np.mean(self.accs_mask_classes), '.4f')
        new_cols.append('task_macro_acc_mask_mean')
        args['task_micro_acc_mask_mean'] = format(np.mean(self.micro_accs_mask_classes), '.4f')
        new_cols.append('task_micro_acc_mask_mean')
        args['auc_mean'] = format(np.mean(self.aucs), '.4f')
        new_cols.append('auc_mean')
        args['multi_class_auc_mean'] = format(np.mean(self.multi_class_auc), '.4f')
        new_cols.append('multi_class_auc_mean')

        # for i, auc in enumerate(self.aucs):
        #     args['task_auc' + str(i + 1)] = auc
        #     new_cols.append('task_auc' + str(i + 1))

        args['forward_transfer_acc'] = self.acc_fwt
        new_cols.append('forward_transfer_acc')
        args['forward_transfer_auc'] = self.auc_fwt
        new_cols.append('forward_transfer_auc')

        args['backward_transfer_acc'] = self.acc_bwt
        new_cols.append('backward_transfer_acc')
        args['backward_transfer_auc'] = self.auc_bwt
        new_cols.append('backward_transfer_auc')

        args['forgetting_acc'] = self.acc_forgetting
        new_cols.append('forgetting_acc')
        args['forgetting_auc'] = self.auc_forgetting
        new_cols.append('forgetting_auc')

        args['acc_results_mask_classes'] = self.acc_results_mask_classes
        new_cols.append('acc_results_mask_classes')
        # args['auc_results_mask_classes'] = self.auc_results_mask_classes
        # new_cols.append('auc_results_mask_classes')

        print(f'task_macro_acc_last:        {self.accs[-1]:.4f}')
        print(f'task_micro_acc_last:        {self.micro_accs[-1]:.4f}')
        print(f'task_macro_acc_mask_last:   {self.accs_mask_classes[-1]:.4f}')
        print(f'task_micro_acc_mask_last:   {self.micro_accs_mask_classes[-1]:.4f}')
        print(f'auc_last:                   {self.aucs[-1]:.4f}')
        print(f'multi_class_auc_last:       {self.multi_class_auc[-1]:.4f}')

        print(f'task_macro_acc_mean:        {np.mean(self.accs):.4f}')
        print(f'task_micro_acc_mean:        {np.mean(self.micro_accs):.4f}')
        print(f'task_macro_acc_mask_mean:   {np.mean(self.accs_mask_classes):.4f}')
        print(f'task_micro_acc_mask_mean:   {np.mean(self.micro_accs_mask_classes):.4f}')
        print(f'auc_mean:                   {np.mean(self.aucs):.4f}')
        print(f'multi_class_auc_mean:       {np.mean(self.multi_class_auc):.4f}')

        print(f'forward_transfer_acc:       {self.acc_fwt:.4f}')
        print(f'forward_transfer_auc:       {self.auc_fwt:.4f}')
        print(f'backward_transfer_acc:      {self.acc_bwt:.4f}')
        print(f'backward_transfer_auc:      {self.auc_bwt:.4f}')
        print(f'forgetting_acc:             {self.acc_forgetting:.4f}')
        print(f'forgetting_auc:             {self.auc_forgetting:.4f}')

        for i, acc in enumerate(self.accs):
            args['task_macro_acc' + str(i + 1)] = acc
            new_cols.append('task_macro_acc' + str(i + 1))

        for i, micro_acc in enumerate(self.micro_accs):
            args['task_micro_acc' + str(i + 1)] = micro_acc
            new_cols.append('task_micro_acc' + str(i + 1))

        for i, acc in enumerate(self.accs_mask_classes):
            args['task_macro_acc_mask' + str(i + 1)] = acc
            new_cols.append('task_macro_acc_mask' + str(i + 1))

        for i, micro_acc in enumerate(self.micro_accs_mask_classes):
            args['task_micro_acc_mask' + str(i + 1)] = micro_acc
            new_cols.append('task_micro_acc_mask' + str(i + 1))
        
        for i, auc in enumerate(self.multi_class_auc):
            args['multi_class_auc' + str(i + 1)] = auc
            new_cols.append('multi_class_auc' + str(i + 1))

        columns = new_cols + columns

        # create_if_not_exists(base_path() + "results/" + self.setting)
        # create_if_not_exists(base_path() + "results/" + self.setting +
        #                      "/" + self.dataset)
        # create_if_not_exists(base_path() + "results/" + self.setting +
        #                      "/" + self.dataset + "/" + self.model)

        # write_headers = False
        # path = base_path() + "results/" + self.setting + "/" + self.dataset\
        #        + "/" + self.model + f"/fold{self.fold}.csv"
        # if not os.path.exists(path):
        #     write_headers = True
        # with open(path, 'a') as tmp:
        #     writer = csv.DictWriter(tmp, fieldnames=columns)
        #     if write_headers:
        #         writer.writeheader()
        #     writer.writerow(args)

        if self.setting == 'class-il':
            # create_if_not_exists(base_path() + "results/task-il/"
            #                      + self.dataset)
            # create_if_not_exists(base_path() + "results/task-il/"
            #                      + self.dataset + "/" + self.model)
            create_if_not_exists(base_path() + self.exp_desc)

            # for i, acc in enumerate(self.accs_mask_classes):
            #     args['task' + str(i + 1)] = acc

            args['forward_transfer_acc'] = self.acc_fwt_mask_classes
            # args['forward_transfer_auc'] = self.auc_fwt_mask_classes
            args['backward_transfer_acc'] = self.acc_bwt_mask_classes
            # args['backward_transfer_auc'] = self.auc_bwt_mask_classes
            args['forgetting_acc'] = self.acc_forgetting_mask_classes
            # args['forgetting_auc'] = self.auc_forgetting_mask_classes

            write_headers = False
            # path = base_path() + "results/task-il" + "/" + self.dataset + "/"\
            #        + self.model + "/mean_accs.csv"
            path = base_path() + self.exp_desc+ f"/fold_{self.fold}.csv"

            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                # import ipdb;ipdb.set_trace()
                # columns = list(args.keys())
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)
