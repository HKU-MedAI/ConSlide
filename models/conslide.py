from utils.buffer import Buffer, Instance_Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from einops import rearrange
import math
import numpy as np
import bisect


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


class ConSlide(ContinualModel):
    NAME = 'conslide'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ConSlide, self).__init__(backbone, loss, args, transform)

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

            loss = self.loss(outputs[0], labels) + 0.00001 * outputs[-1].mean()
            

            if not self.buffer.is_empty():
                t = np.random.randint(task)
                size = np.random.randint(100, 250)
                buf_inputs00, buf_inputs01, buf_inputs10, buf_inputs11 = self.buffer.get_data(size=size, task=t)
                buf_outputs0 = self.net([buf_inputs00, buf_inputs01])
                buf_outputs1 = self.net([buf_inputs10, buf_inputs11])
                loss += self.args.alpha * self.loss(buf_outputs0[0], torch.tensor([2*t]).to(self.device))
                loss += self.args.alpha * self.loss(buf_outputs1[0], torch.tensor([2*t+1]).to(self.device))

            loss.backward()
            self.opt.step()

        return loss.item()
    

    def save_buffer(self, inputs0, inputs1, labels, task):

        self.opt.zero_grad()
        if inputs0.shape[0] > 1:
            with torch.no_grad():
                outputs = self.net([inputs0, inputs1])
                
                # Add instance-level feat to buffer
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

                self.buffer.add_data(examples=[topk_inputs0, topk_inputs1], labels=labels, task_labels=task)
