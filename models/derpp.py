# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, default=0.2,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, default=0.2,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs0, inputs1, labels, task, ssl=False):
        if task == 0 and ssl:
            self.opt.zero_grad()
            outputs = self.net([inputs0, inputs1])
            loss = 0.001 * outputs[-1].mean()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            outputs = self.net([inputs0, inputs1])
            # import ipdb;ipdb.set_trace()
            # loss = self.loss(outputs[0], labels) + 0.000001 * outputs[-1].mean()
            loss = self.loss(outputs[0], labels)
            # loss = self.loss(outputs[0], labels)

            if not self.buffer.is_empty():
                # import ipdb;ipdb.set_trace()
                buf_inputs, _, buf_logits = self.buffer.get_data()
                buf_outputs = self.net([buf_inputs[0], buf_inputs[1]])
                # import ipdb;ipdb.set_trace()
                loss += self.args.alpha * F.mse_loss(buf_outputs[0], buf_logits)

                buf_inputs, buf_labels, _ = self.buffer.get_data()
                buf_outputs = self.net([buf_inputs[0], buf_inputs[1]])
                loss += self.args.beta * self.loss(buf_outputs[0], buf_labels)

            loss.backward()
            self.opt.step()

            if self.args.buffer_size != 0:
                self.buffer.add_data(examples=[inputs0, inputs1],
                                    labels=labels,
                                    logits=outputs[0])

        return loss.item()
