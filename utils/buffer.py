# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from copy import deepcopy

def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """
        
    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)
        
        data_concatenate = torch.cat if type(dataset.train_loader.dataset.data) == torch.Tensor else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            refold_transform = lambda x: x.cpu()
        else:    
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                refold_transform = lambda x: (x.cpu()*255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                refold_transform = lambda x: (x.cpu()*255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
            ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
            ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
                ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
                ])

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            # import ipdb;ipdb.set_trace()
            if attr is not None and not hasattr(self, attr_str):
                # typ = torch.int64 if attr_str.endswith('els') else torch.float32
                # setattr(self, attr_str, torch.zeros((self.buffer_size,
                #         *attr.shape[1:]), dtype=typ, device=self.device))
                setattr(self, attr_str, list())

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        # import ipdb;ipdb.set_trace()
        examples = [i.to(self.device) for i in examples]
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        # for i in range(examples[0].shape[0]):
        index = reservoir(self.num_seen_examples, self.buffer_size)
        # self.num_seen_examples += 1
        if index >= 0:
            if index != self.num_seen_examples:
                self.examples[index] = examples
                # self.examples1[index] = examples[1]
                if labels is not None:
                    self.labels[index] = labels.to(self.device)
                if logits is not None:
                    self.logits[index] = logits.detach().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels.to(self.device)
            else:
                # import ipdb;ipdb.set_trace()
                self.examples.append(examples)
                if labels is not None:
                    self.labels.append(labels.to(self.device))
                if logits is not None:
                    self.logits.append(logits.detach().to(self.device))
                if task_labels is not None:
                    self.task_labels.append(task_labels.to(self.device))
        self.num_seen_examples += 1

    def get_data(self, size=None, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        # if size > min(self.num_seen_examples, len(self.examples)):
        #     size = min(self.num_seen_examples, len(self.examples))

        choice = np.random.choice(min(self.num_seen_examples, len(self.examples)),
                                  size=1, replace=False)
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples[choice]]).to(self.device),)
        # import ipdb;ipdb.set_trace()
        ret_tuple = ((self.examples[choice[0]]), )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice[0]],)
        # import ipdb;ipdb.set_trace()
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

        return ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


# class Instance_Buffer:
#     """
#     The memory buffer of rehearsal method.
#     """
#     def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
#         assert mode in ['ring', 'reservoir']
#         self.buffer_size = buffer_size
#         self.device = device
#         self.num_seen_examples = [0]*n_tasks
#         self.functional_index = eval(mode)
#         if mode == 'ring':
#             assert n_tasks is not None
#             self.task_number = n_tasks
#             self.buffer_portion_size = buffer_size // n_tasks
#         self.attributes = ['examples0', 'examples1', 'labels']
#         self.examples0 = []
#         self.examples1 = []
#         self.labels = []
#         self.seen_task = []

#     def to(self, device):
#         self.device = device
#         for attr_str in self.attributes:
#             if hasattr(self, attr_str):
#                 setattr(self, attr_str, getattr(self, attr_str).to(device))
#         return self

#     def __len__(self):
#         return min(self.num_seen_examples, self.buffer_size)


#     def init_tensors(self, examples0: torch.Tensor, examples1: torch.Tensor, labels: torch.Tensor,
#                      task_labels: torch.Tensor) -> None:
#         """
#         Initializes just the required tensors.
#         :param examples: tensor containing the images
#         :param labels: tensor containing the labels
#         :param logits: tensor containing the outputs of the network
#         :param task_labels: tensor containing the task labels
#         """
#         # torch.int64 if attr_str.endswith('els') else torch.float32
#         self.examples0.append(torch.zeros((self.buffer_size,*examples0.shape[1:]), dtype=torch.float32, device=self.device))
#         self.examples1.append(torch.zeros((self.buffer_size,*examples1.shape[1:]), dtype=torch.float32, device=self.device))
#         self.labels.append(torch.zeros((self.buffer_size,*labels.shape[1:]), dtype=torch.int64, device=self.device))

#         # for attr_str in self.attributes:
#         #     attr = eval(attr_str)
#         #     attr_str = attr_str + f'_task{task_labels}'
#         #     import ipdb;ipdb.set_trace()
#         #     if attr is not None and not hasattr(self, attr_str):
#         #         typ = torch.int64 if attr_str.endswith('els') else torch.float32
#         #         # setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))
#         #         self.examples0.append(torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))
#         #         # setattr(self, attr_str, list())

#     def add_data(self, examples, labels=None, logits=None, task_labels=None):
#         """
#         Adds the data to the memory buffer according to the reservoir strategy.
#         :param examples: tensor containing the images
#         :param labels: tensor containing the labels
#         :param logits: tensor containing the outputs of the network
#         :param task_labels: tensor containing the task labels
#         :return:
#         """
#         # import ipdb;ipdb.set_trace()
#         examples = [i.to(self.device) for i in examples]
#         if (not hasattr(self, 'examples0')) or (task_labels not in self.seen_task):
#             self.seen_task.append(task_labels)
#             self.init_tensors(examples[0], examples[1], labels, task_labels)

#         for i in range(examples[0].shape[0]):
#             index = reservoir(self.num_seen_examples[task_labels], self.buffer_size)
#             # self.num_seen_examples += 1
#             if index >= 0:
#                 # if index != self.num_seen_examples:
#                 #     self.examples[index] = examples[0][i]
#                 #     # self.examples1[index] = examples[1]
#                 #     # if labels is not None:
#                 #     #     self.labels[index] = labels.to(self.device)
#                 #     if logits is not None:
#                 #         self.logits[index] = logits[i].detach().to(self.device)
#                 #     if task_labels is not None:
#                 #         self.task_labels[index] = task_labels.to(self.device)
#                 # else:
#                 #     self.examples.append(examples[0][i])
#                 #     # if labels is not None:
#                 #     #     self.labels.append(labels.to(self.device))
#                 #     if logits is not None:
#                 #         self.logits.append(logits[i].detach().to(self.device))
#                 #     if task_labels is not None:
#                 #         self.task_labels.append(task_labels.to(self.device))
#                 # import ipdb;ipdb.set_trace()
#                 self.examples0[task_labels][index] = examples[0][i].to(self.device)
#                 self.examples1[task_labels][index] = examples[1][i].to(self.device)
#                 self.labels[task_labels][index] = labels[0].to(self.device)
#                 # if labels is not None:
#                 #     self.labels[index] = labels[i].to(self.device)
#                 # if logits is not None:
#                 #     self.logits[index] = logits[i].detach().to(self.device)
#                 # if task_labels is not None:
#                 #     self.task_labels[index] = task_labels[i].to(self.device)

#             self.num_seen_examples[task_labels] += 1

#     def get_data(self, size=None, transform: transforms=None, return_index=False, task=None) -> Tuple:
#         """
#         Random samples a batch of size items.
#         :param size: the number of requested items
#         :param transform: the transformation to be applied (data augmentation)
#         :return:
#         """
#         # if size > min(self.num_seen_examples[task], len(self.examples0[task])):
#         #     size = min(self.num_seen_examples[task], len(self.examples0[task]))
#         size0, size1 = size, size
#         if size0 > min(self.num_seen_examples[task], sum(self.labels[task] == task*2).cpu()):
#             size0 = min(self.num_seen_examples[task], sum(self.labels[task] == task*2).cpu())
#         if size1 > min(self.num_seen_examples[task], sum(self.labels[task] == task*2 + 1).cpu()):
#             size1 = min(self.num_seen_examples[task], sum(self.labels[task] == task*2 + 1).cpu())

#         # choice0 = np.random.choice(min(self.num_seen_examples[task], len(self.examples0[task])),
#         #                           size=size, replace=False)
#         # import ipdb;ipdb.set_trace()
#         choice0 = np.random.choice(min(self.num_seen_examples[task], sum(self.labels[task] == task*2).cpu()),
#                                   size=size0, replace=False)
#         choice1 = np.random.choice(min(self.num_seen_examples[task], sum(self.labels[task] == task*2 + 1).cpu()),
#                                   size=size1, replace=False)
#         # if transform is None: transform = lambda x: x
#         # ret_tuple = (torch.stack([transform(ee.cpu())
#         #                     for ee in self.examples[choice]]).to(self.device),)
#         # import ipdb;ipdb.set_trace()
#         ret_tuple = (
#             (self.examples0[task][self.labels[task] == task*2][choice0]),
#             (self.examples1[task][self.labels[task] == task*2][choice0]),
#             (self.examples0[task][self.labels[task] == task*2+1][choice1]),
#             (self.examples1[task][self.labels[task] == task*2+1][choice1]),
#             )
#         # for attr_str in self.attributes[1:]:
#         #     if hasattr(self, attr_str):
#         #         attr = getattr(self, attr_str)
#         #         ret_tuple += (attr[choice],)
#         # import ipdb;ipdb.set_trace()
#         # if not return_index:
#         #     return ret_tuple
#         # else:
#         #     return (torch.tensor(choice).to(self.device), ) + ret_tuple

#         return ret_tuple

#     def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
#         """
#         Returns the data by the given index.
#         :param index: the index of the item
#         :param transform: the transformation to be applied (data augmentation)
#         :return:
#         """
#         if transform is None: transform = lambda x: x
#         ret_tuple = (torch.stack([transform(ee.cpu())
#                             for ee in self.examples[indexes]]).to(self.device),)
#         for attr_str in self.attributes[1:]:
#             if hasattr(self, attr_str):
#                 attr = getattr(self, attr_str).to(self.device)
#                 ret_tuple += (attr[indexes],)
#         return ret_tuple


#     def is_empty(self) -> bool:
#         """
#         Returns true if the buffer is empty, false otherwise.
#         """
#         if max(self.num_seen_examples) == 0:
#             return True
#         else:
#             return False

#     def get_all_data(self, transform: transforms=None) -> Tuple:
#         """
#         Return all the items in the memory buffer.
#         :param transform: the transformation to be applied (data augmentation)
#         :return: a tuple with all the items in the memory buffer
#         """
#         if transform is None: transform = lambda x: x
#         ret_tuple = (torch.stack([transform(ee.cpu())
#                             for ee in self.examples]).to(self.device),)
#         for attr_str in self.attributes[1:]:
#             if hasattr(self, attr_str):
#                 attr = getattr(self, attr_str)
#                 ret_tuple += (attr,)
#         return ret_tuple

#     def empty(self) -> None:
#         """
#         Set all the tensors to None.
#         """
#         for attr_str in self.attributes:
#             if hasattr(self, attr_str):
#                 delattr(self, attr_str)
#         self.num_seen_examples = 0

class Instance_Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples0', 'examples1', 'labels']
        # self.examples0 = []
        # self.examples1 = []
        # self.labels = []
        self.seen_task = []

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples0: torch.Tensor, examples1: torch.Tensor, labels: torch.Tensor,
                     task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        # torch.int64 if attr_str.endswith('els') else torch.float32
        self.examples0 = torch.zeros((self.buffer_size,*examples0.shape[1:]), dtype=torch.float32, device=self.device)
        self.examples1 = torch.zeros((self.buffer_size,*examples1.shape[1:]), dtype=torch.float32, device=self.device)
        self.labels = torch.ones((self.buffer_size,*labels.shape[1:]), dtype=torch.int64, device=self.device) * -1

        # for attr_str in self.attributes:
        #     attr = eval(attr_str)
        #     attr_str = attr_str + f'_task{task_labels}'
        #     import ipdb;ipdb.set_trace()
        #     if attr is not None and not hasattr(self, attr_str):
        #         typ = torch.int64 if attr_str.endswith('els') else torch.float32
        #         # setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))
        #         self.examples0.append(torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))
        #         # setattr(self, attr_str, list())

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        # import ipdb;ipdb.set_trace()
        examples = [i.to(self.device) for i in examples]
        # if (not hasattr(self, 'examples0')) or (task_labels not in self.seen_task):
        if not hasattr(self, 'examples0'):
            self.seen_task.append(task_labels)
            self.init_tensors(examples[0], examples[1], labels, task_labels)

        for i in range(examples[0].shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            # self.num_seen_examples += 1
            if index >= 0:
                # if index != self.num_seen_examples:
                #     self.examples[index] = examples[0][i]
                #     # self.examples1[index] = examples[1]
                #     # if labels is not None:
                #     #     self.labels[index] = labels.to(self.device)
                #     if logits is not None:
                #         self.logits[index] = logits[i].detach().to(self.device)
                #     if task_labels is not None:
                #         self.task_labels[index] = task_labels.to(self.device)
                # else:
                #     self.examples.append(examples[0][i])
                #     # if labels is not None:
                #     #     self.labels.append(labels.to(self.device))
                #     if logits is not None:
                #         self.logits.append(logits[i].detach().to(self.device))
                #     if task_labels is not None:
                #         self.task_labels.append(task_labels.to(self.device))
                # import ipdb;ipdb.set_trace()
                self.examples0[index] = examples[0][i].to(self.device)
                self.examples1[index] = examples[1][i].to(self.device)
                self.labels[index] = labels[0].to(self.device)
                # if labels is not None:
                #     self.labels[index] = labels[i].to(self.device)
                # if logits is not None:
                #     self.logits[index] = logits[i].detach().to(self.device)
                # if task_labels is not None:
                #     self.task_labels[index] = task_labels[i].to(self.device)

            self.num_seen_examples += 1

    def get_data(self, size=None, transform: transforms=None, return_index=False, task=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        # if size > min(self.num_seen_examples[task], len(self.examples0[task])):
        #     size = min(self.num_seen_examples[task], len(self.examples0[task]))
        size0, size1 = size, size
        if size0 > min(self.num_seen_examples, sum(self.labels == task*2).cpu()):
            size0 = min(self.num_seen_examples, sum(self.labels == task*2).cpu()).item()
        if size1 > min(self.num_seen_examples, sum(self.labels == task*2 + 1).cpu()):
            size1 = min(self.num_seen_examples, sum(self.labels == task*2 + 1).cpu()).item()

        # choice0 = np.random.choice(min(self.num_seen_examples[task], len(self.examples0[task])),
        #                           size=size, replace=False)
        # import ipdb;ipdb.set_trace()
        choice0 = np.random.choice(min(self.num_seen_examples, sum(self.labels == task*2).cpu()),
                                  size=size0, replace=False)
        # print(min(self.num_seen_examples, sum(self.labels == task*2 + 1).cpu()))
        # print(size1)
        # if min(self.num_seen_examples, sum(self.labels == task*2 + 1).cpu()) == 0:
        #     import ipdb;ipdb.set_trace()
        choice1 = np.random.choice(min(self.num_seen_examples, sum(self.labels == task*2 + 1).cpu()),
                                  size=size1, replace=False)
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples[choice]]).to(self.device),)
        # import ipdb;ipdb.set_trace()
        ret_tuple = (
            (self.examples0[self.labels == task*2][choice0]),
            (self.examples1[self.labels == task*2][choice0]),
            (self.examples0[self.labels == task*2+1][choice1]),
            (self.examples1[self.labels == task*2+1][choice1]),
            )
        # for attr_str in self.attributes[1:]:
        #     if hasattr(self, attr_str):
        #         attr = getattr(self, attr_str)
        #         ret_tuple += (attr[choice],)
        # import ipdb;ipdb.set_trace()
        # if not return_index:
        #     return ret_tuple
        # else:
        #     return (torch.tensor(choice).to(self.device), ) + ret_tuple

        return ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

class task_Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            # import ipdb;ipdb.set_trace()
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
                # setattr(self, attr_str, list())

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        # import ipdb;ipdb.set_trace()
        examples = [i.to(self.device) for i in examples]
        if not hasattr(self, 'examples'):
            self.init_tensors(examples[0], labels, logits, task_labels)

        for i in range(examples[0].shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            # self.num_seen_examples += 1
            if index >= 0:
                # if index != self.num_seen_examples:
                #     self.examples[index] = examples[0][i]
                #     # self.examples1[index] = examples[1]
                #     # if labels is not None:
                #     #     self.labels[index] = labels.to(self.device)
                #     if logits is not None:
                #         self.logits[index] = logits[i].detach().to(self.device)
                #     if task_labels is not None:
                #         self.task_labels[index] = task_labels.to(self.device)
                # else:
                #     self.examples.append(examples[0][i])
                #     # if labels is not None:
                #     #     self.labels.append(labels.to(self.device))
                #     if logits is not None:
                #         self.logits.append(logits[i].detach().to(self.device))
                #     if task_labels is not None:
                #         self.task_labels.append(task_labels.to(self.device))
                self.examples[index] = examples[0][i].to(self.device)
                # if labels is not None:
                #     self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].detach().to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

            self.num_seen_examples += 1

    def get_data(self, size=None, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, len(self.examples)):
            size = min(self.num_seen_examples, len(self.examples))

        choice = np.random.choice(min(self.num_seen_examples, len(self.examples)),
                                  size=size, replace=False)
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples[choice]]).to(self.device),)
        # import ipdb;ipdb.set_trace()
        ret_tuple = ((self.examples[choice]), )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        # import ipdb;ipdb.set_trace()
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

        return ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
