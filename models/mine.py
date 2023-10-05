# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer, Instance_Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from einops import rearrange
import math
import numpy as np
import bisect
from info_nce import InfoNCE

infonce_loss = InfoNCE(negative_mode='unpaired')

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


class Mine(ContinualModel):
    NAME = 'mine'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Mine, self).__init__(backbone, loss, args, transform)

        self.buffer = Instance_Buffer(self.args.buffer_size, self.device, n_tasks=4)

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
            # loss = self.loss(outputs[0], labels)
            loss = self.loss(outputs[0], labels) + 0.00001 * outputs[-1].mean()
            
            # loss_inst = 0
            # for _ in range(120):
            # if not self.buffer.is_empty():
            #     buf_inputs, buf_logits = self.buffer.get_data(size=200, task=0)
            #     buf_outputs = self.net(buf_inputs, inst_level=True)
            #     # import ipdb;ipdb.set_trace()
            #     loss += self.args.alpha * F.mse_loss(buf_outputs[0], buf_logits)

            if not self.buffer.is_empty():
                # for t in range(task):
                # for _ in range(3):
                t = np.random.randint(task)
                size = np.random.randint(100, 250)
                buf_inputs00, buf_inputs01, buf_inputs10, buf_inputs11 = self.buffer.get_data(size=size, task=t)
                # import ipdb;ipdb.set_trace()
                buf_outputs0 = self.net([buf_inputs00, buf_inputs01])
                buf_outputs1 = self.net([buf_inputs10, buf_inputs11])
                loss += self.args.alpha * self.loss(buf_outputs0[0], torch.tensor([2*t]).to(self.device))
                loss += self.args.alpha * self.loss(buf_outputs1[0], torch.tensor([2*t+1]).to(self.device))
                # import ipdb;ipdb.set_trace()
                # loss += self.args.beta * infonce_loss(buf_outputs0[0], buf_outputs1[0], outputs[0])
                                            # infonce_loss(buf_outputs0[0], buf_outputs0[0], torch.cat((outputs[0], buf_outputs1[0]))) +
                                            # infonce_loss(buf_outputs1[0], buf_outputs1[0], torch.cat((outputs[0], buf_outputs0[0]))))

                # import ipdb;ipdb.set_trace()
                # loss += self.args.alpha * F.mse_loss(buf_outputs[0], buf_logits)

            # loss += loss_inst / 120
            loss.backward()
            self.opt.step()
        
        # # Add instance-level feat to buffer
        # patch_attention_score = outputs[-1]
        # region_attention_score = torch.sum(patch_attention_score, dim=-1)
        # # import ipdb;ipdb.set_trace()
        # topk_idx = torch.topk(region_attention_score, math.ceil(0.1 * region_attention_score.shape[0]))[1]
        # topk_inputs0 = inputs0[topk_idx]
        # topk_inputs1 = inputs1[topk_idx]
        # inst_att = patch_attention_score[topk_idx]
        # with torch.no_grad():
        #     topk_logits = self.net(topk_inputs0, returnt='inst_feat', inst_att=inst_att)
        # # import ipdb;ipdb.set_trace()
        # self.buffer.add_data(examples=[topk_inputs0, topk_inputs1],
        #                      logits=topk_logits)

        return loss.item()
    
    # def save_buffer(self, inputs0, inputs1, labels, task):

    #     self.opt.zero_grad()
    #     if inputs0.shape[0] > 1:
    #         with torch.no_grad():
    #             outputs = self.net([inputs0, inputs1])
                
    #             # Add instance-level feat to buffer
    #             import ipdb;ipdb.set_trace()
    #             patch_attention_score = outputs[-1]
    #             region_attention_score = torch.sum(patch_attention_score, dim=-1)
    #             range_idx = np.arange(0, 1, 1 / region_attention_score.shape[0])

    #             if 1:
    #                 att_sum = 0
    #                 cumu_att = []
    #                 for i in range(region_attention_score.shape[0]):
    #                     att_sum += region_attention_score[i]
    #                     cumu_att.append(float(att_sum.cpu()))
    #                 samp_idx = []
    #                 for j in range(range_idx.shape[0]):
    #                     tmp = bisect.bisect(cumu_att, range_idx[j])
    #                     if tmp <= region_attention_score.shape[0]-1:
    #                         samp_idx.append(tmp)
    #                 samp_idx = list(set(samp_idx))
    #                 if len(samp_idx) == 0:
    #                     samp_idx = torch.topk(region_attention_score, math.ceil(0.2 * region_attention_score.shape[0]))[1]
    #             else:
    #                 samp_idx = torch.topk(region_attention_score, math.ceil(0.2 * region_attention_score.shape[0]))[1]
    #             topk_inputs0 = inputs0[samp_idx]
    #             topk_inputs1 = inputs1[samp_idx]
    #             inst_att = patch_attention_score[samp_idx]
    #             # with torch.no_grad():
    #             topk_logits = self.net(topk_inputs0, returnt='inst_feat', inst_att=inst_att)
    #             # import ipdb;ipdb.set_trace()
    #             self.buffer.add_data(examples=[topk_inputs0, topk_inputs1],
    #                                 logits=topk_logits)
    def save_buffer(self, inputs0, inputs1, labels, task):

        self.opt.zero_grad()
        if inputs0.shape[0] > 1:
            with torch.no_grad():
                outputs = self.net([inputs0, inputs1])
                
                # Add instance-level feat to buffer
                # import ipdb;ipdb.set_trace()
                region_attention_score = outputs[-2].squeeze()
                region_attention_score = 1 / region_attention_score
                region_attention_score = region_attention_score / sum(region_attention_score)
                range_idx = np.arange(0, 1, 1 / (0.2 * region_attention_score.shape[0]))

                if 0:
                    att_sum = 0
                    cumu_att = []
                    for i in range(region_attention_score.shape[0]):
                        att_sum += region_attention_score[i]
                        cumu_att.append(float(att_sum.cpu()))
                    samp_idx = []
                    for j in range(range_idx.shape[0]):
                        tmp = bisect.bisect(cumu_att, range_idx[j])
                        if tmp <= region_attention_score.shape[0]-1:
                            samp_idx.append(tmp)
                    samp_idx = list(set(samp_idx))
                    if len(samp_idx) == 0:
                        samp_idx = torch.topk(region_attention_score, math.ceil(0.2 * region_attention_score.shape[0]))[1]
                elif 0:
                    samp_idx = torch.topk(region_attention_score, math.ceil(0.2 * region_attention_score.shape[0]))[1]
                else:
                    samp_idx = np.random.randint(inputs0.shape[0], size=math.ceil(0.2 * region_attention_score.shape[0]))
                topk_inputs0 = inputs0[samp_idx]
                topk_inputs1 = inputs1[samp_idx]
                # with torch.no_grad():
                # import ipdb;ipdb.set_trace()
                # topk_logits = self.net(topk_inputs0, returnt='inst_feat')
                # import ipdb;ipdb.set_trace()
                self.buffer.add_data(examples=[topk_inputs0, topk_inputs1], labels=labels, task_labels=task)

