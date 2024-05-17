#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 20:17
# @Author  : yulong
# @File    : model.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as INIT
import dgl.backend as F
import torch.nn.functional as torch_F

import scipy as sp
from tqdm import tqdm

from dgl.base import NID, EID
import dgl
import os

from collections import defaultdict
from utils.loss import *
from utils.utils import *

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from dataloader import TestDataset
from dataloader import *

norm = lambda x, p: x.norm(p=p) ** p
get_scalar = lambda x: x.detach().item()


class KGEModel(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, device,
                 double_entity_embedding=False, double_relation_embedding=False,triple_relation_embedding=False, triple_entity_embedding=False):
        super(KGEModel, self).__init__()
        self.args = args
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.device = device
        self.epsilon = 2.0
        self.u = 1.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 5, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim, momentum=0.5),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim, hidden_dim // 4),
            torch.nn.BatchNorm1d(hidden_dim // 4, momentum=0.5),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 4, hidden_dim // 8),
            torch.nn.BatchNorm1d(hidden_dim // 8, momentum=0.5),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 8, 2),
            torch.nn.Softmax(dim=1)
        )
        self.mlp.to(device)

        self.mlp_loss_func = nn.CrossEntropyLoss()
        self.mlp_loss_func.to(device)

        #self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        #self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        #self.entity_dim = hidden_dim * 3 if triple_entity_embedding else hidden_dim        
        #self.relation_dim = hidden_dim * 3 if triple_relation_embedding else hidden_dim
        
        self.entity_embedding=hidden_dim*3 if triple_entity_embedding else (hidden_dim*2 if double_entity_embedding else hidden_dim)
        self.relation_embedding=hidden_dim*3 if triple_relation_embedding else (hidden_dim*2 if double_relation_embedding else hidden_dim)




        self.embedding_range = (self.gamma.item() + self.epsilon) / hidden_dim

        self.entity_embedding = SparseEmbedding(args, nentity, self.entity_dim, device)
        self.relation_embedding = SparseEmbedding(args, nrelation, self.relation_dim, device)

        self.reset_parameters()

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'RotatEv2', 'pRotatE', 'TripleRE', 'AutoSF', 'pairRE','InterHT']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, g, mode, gpu_id=-1, trace=False, neg_deg_sample=False):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            g.ndata['emb'] = self.entity_embedding(g.ndata['id'], gpu_id, trace)
            g.edata['emb'] = self.relation_embedding(g.edata['id'], gpu_id, trace)

            head_ids, tail_ids = g.all_edges(order='eid')
            head = g.ndata['emb'][head_ids].unsqueeze(1)
            rel = g.edata['emb'].unsqueeze(1)
            tail = g.ndata['emb'][tail_ids].unsqueeze(1)

            batch_size = head.size()[0]

            # mlp_inp = torch.cat((head.view(batch_size, self.entity_dim), rel.view(batch_size, self.relation_dim)), dim=1)
            # mlp_inp = torch.cat((mlp_inp, tail.view(batch_size, self.entity_dim)), dim=1)

        elif mode == 'head-batch-test':
            pos_g, neg_g = g
            num_chunks = neg_g.num_chunks
            chunk_size = neg_g.chunk_size
            neg_sample_size = neg_g.neg_sample_size
            mask = F.ones((num_chunks, chunk_size * (neg_sample_size + chunk_size)),
                          dtype=F.float32, ctx=F.context(pos_g.ndata['emb']))

            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            neg_head = self.entity_embedding(neg_head_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            tail = pos_g.ndata['emb'][tail_ids].unsqueeze(1)
            rel = pos_g.edata['emb'].unsqueeze(1)
            batch_size = tail.size()[0]

            if neg_deg_sample:
                head = pos_g.ndata['emb'][head_ids]
                head = head.reshape(num_chunks, chunk_size, -1)
                neg_head = neg_head.reshape(num_chunks, neg_sample_size, -1)
                neg_head = F.cat([head, neg_head], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:,0::(neg_sample_size + 1)] = 0
            # neg_head = neg_head.reshape(num_chunks * neg_sample_size, -1)
            # head = neg_head.unsqueeze(0).repeat([batch_size, 1, 1])
            head = neg_head.unsqueeze(1)
            # mlp_inp = torch.cat((head.view(batch_size, self.entity_dim), rel.view(batch_size, self.relation_dim)), dim=1)
            # mlp_inp = torch.cat((mlp_inp, tail.view(batch_size, self.entity_dim)), dim=1)

        elif mode == 'tail-batch-test':
            pos_g, neg_g = g
            num_chunks = neg_g.num_chunks
            chunk_size = neg_g.chunk_size
            neg_sample_size = neg_g.neg_sample_size
            mask = F.ones((num_chunks, chunk_size * (neg_sample_size + chunk_size)),
                          dtype=F.float32, ctx=F.context(pos_g.ndata['emb']))

            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_embedding(neg_tail_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            head = pos_g.ndata['emb'][head_ids].unsqueeze(1)
            rel = pos_g.edata['emb'].unsqueeze(1)
            batch_size = head.size()[0]

            # This is negative edge construction similar to the above.
            if neg_deg_sample:
                tail = pos_g.ndata['emb'][tail_ids]
                tail = tail.reshape(num_chunks, chunk_size, -1)
                neg_tail = neg_tail.reshape(num_chunks, neg_sample_size, -1)
                neg_tail = F.cat([tail, neg_tail], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0
            # neg_tail = neg_tail.reshape(num_chunks * neg_sample_size, -1)
            # tail = neg_tail.unsqueeze(0).repeat([batch_size, 1, 1])
            tail = neg_tail.unsqueeze(1)
            # mlp_inp = torch.cat((head.view(batch_size, self.entity_dim), rel.view(batch_size, self.relation_dim)), dim=1)
            # mlp_inp = torch.cat((mlp_inp, tail.view(batch_size, self.entity_dim)), dim=1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotatEv2': self.RotatEv2,
            'pRotatE': self.pRotatE,
            'TripleRE': self.TripleRE,
            'AutoSF': self.AutoSF,
            'pairRE': self.PairRE,
            'InterHT':self.InterHT
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, rel, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        if neg_deg_sample:
            neg_g.neg_sample_size = neg_sample_size
            mask = mask.reshape(num_chunks, chunk_size, neg_sample_size)
            return score * mask, None #mlp_inp
        else:
            return score, None #mlp_inp

    def forward_test(self, headId, relaId, tailId, gpu_id=-1):
        head = self.entity_embedding(headId, gpu_id, False).unsqueeze(1)
        tail = self.entity_embedding(tailId, gpu_id, False).unsqueeze(1)
        rel = self.relation_embedding(relaId, gpu_id, False).unsqueeze(1)

        batch_size = head.size()[0]

        mlp_inp = torch.cat((head.view(batch_size, self.entity_dim), rel.view(batch_size, self.relation_dim)), dim=1)
        mlp_inp = torch.cat((mlp_inp, tail.view(batch_size, self.entity_dim)), dim=1)

        preds = self.mlp(mlp_inp)

        return preds

    def reset_parameters(self):
        self.entity_embedding.init(self.embedding_range)
        self.relation_embedding.init(self.embedding_range)

    def update(self):
        """ Update the embeddings in the model

        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        gpu_id = self.args.gpu_id
        self.entity_embedding.update(gpu_id)
        self.relation_embedding.update(gpu_id)

    def transformer(self):
        pass

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = torch_F.normalize(head, 2, -1)
        tail = torch_F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

#    def TripleRE(self, head, relation, tail, mode):
#        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)
#
#        head = torch_F.normalize(head, 2, -1)
#        tail = torch_F.normalize(tail, 2, -1)
#
#       score = head * re_head - tail * re_tail + re_mid
#        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
#        return score

    def TripleRE(self, head, relation, tail, mode):
        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)

        head = torch_F.normalize(head, 2, -1)
        tail = torch_F.normalize(tail, 2, -1)

        e_h = torch.ones_like(re_head)
        e_t = torch.ones_like(re_tail)

        score = head * (re_head + self.u*e_h) - tail * (re_tail + self.u*e_t) + self.u*re_mid
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
        



    
    def InterHT(self, head, relation, tail, mode):
        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)
        e_h = torch.ones_like(head)
        e_t = torch.ones_like(tail)

        head = torch_F.normalize(head, 2, -1)
        tail = torch_F.normalize(tail, 2, -1)

        score = self.u * head * tail + head * (
            self.u * re_head + e_h) - tail * (self.u * re_tail + e_t) + re_mid

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

#    def InterHT(self, head, relation, tail, mode):
#        a_head, b_head = torch.chunk(head, 2, dim=2)
#        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)
#        a_tail, b_tail = torch.chunk(tail, 2, dim=2)
#
#        e_h = torch.ones_like(b_head)
#        e_t = torch.ones_like(b_tail)
#
#        a_head = torch_F.normalize(a_head, 2, -1)
#        a_tail = torch_F.normalize(a_tail, 2, -1)
#        b_head = torch_F.normalize(b_head, 2, -1)
#        b_tail = torch_F.normalize(b_tail, 2, -1)
#        b_head = b_head + self.u * e_h
#        b_tail = b_tail + self.u * e_t
#
#        score = a_head * b_tail - a_tail * b_head + re_mid
#        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
#        return score



    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        # phase_relation = relation / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def RotatEv2(self, head, relation, tail, mode, r_norm=None):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=2)
        im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=2)

        re_score_head = re_head * re_relation_head - im_head * im_relation_head
        im_score_head = re_head * im_relation_head + im_head * re_relation_head

        re_score_tail = re_tail * re_relation_tail - im_tail * im_relation_tail
        im_score_tail = re_tail * im_relation_tail + im_tail * re_relation_tail

        re_score = re_score_head - re_score_tail
        im_score = im_score_head - im_score_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def AutoSF(self, head, relation, tail, mode):
        if mode == 'head-batch':
            rs = torch.chunk(relation, 4, dim=-1)
            ts = torch.chunk(tail, 4, dim=-1)
            rt0 = rs[0] * ts[0]
            rt1 = rs[1] * ts[1] + rs[2] * ts[3]
            rt2 = rs[0] * ts[2] + rs[2] * ts[3]
            rt3 = -rs[1] * ts[1] + rs[3] * ts[2]
            rts = torch.cat([rt0, rt1, rt2, rt3], dim=-1)
            score = torch.sum(head * rts, dim=-1)

        else:
            hs = torch.chunk(head, 4, dim=-1)
            rs = torch.chunk(relation, 4, dim=-1)
            hr0 = hs[0] * rs[0]
            hr1 = hs[1] * rs[1] - hs[3] * rs[1]
            hr2 = hs[2] * rs[0] + hs[3] * rs[3]
            hr3 = hs[1] * rs[2] + hs[2] * rs[2]
            hrs = torch.cat([hr0, hr1, hr2, hr3], dim=-1)
            score = torch.sum(hrs * tail, dim=-1)

        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range / pi)
        phase_relation = relation / (self.embedding_range / pi)
        phase_tail = tail / (self.embedding_range / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        pos_g, neg_g = next(train_iterator)

        mode = "tail-batch-test"
        if neg_g.neg_head:
            mode = "head-batch-test"

        positive_score, pos_mlp_inp = model.forward(pos_g, mode="single", gpu_id=-1, trace=True, neg_deg_sample=False)
        negative_score, neg_mlp_inp = model.forward([pos_g, neg_g], mode=mode, gpu_id=-1, trace=True, neg_deg_sample=False)

        if args.do_downstream:
            positive_label = torch.ones_like(positive_score.squeeze(), dtype=torch.int64)
            positive_preds = model.mlp(pos_mlp_inp)
            positive_mlp_loss = model.mlp_loss_func(positive_preds, positive_label)

            negative_label = torch.zeros_like(negative_score.squeeze(), dtype=torch.int64)
            negative_preds = model.mlp(neg_mlp_inp)
            negative_mlp_loss = model.mlp_loss_func(negative_preds, negative_label)

            positive_mlp_loss = positive_mlp_loss.mean()
            negative_mlp_loss = negative_mlp_loss.mean()

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (torch_F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * torch_F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = torch_F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = torch_F.logsigmoid(positive_score).squeeze(dim=1)

        # loss, loss_log = model.loss_gen.get_total_loss(positive_score, negative_score, edge_weight=None)

        # if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        # else:
        #     positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        #     negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        if args.do_downstream:
            loss = (positive_sample_loss + negative_sample_loss) / 2 + (positive_mlp_loss + negative_mlp_loss) / 2
        else:
            loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.emb.norm(p=3) ** 3 +
                    model.relation_embedding.emb.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': get_scalar(regularization)}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        model.update()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            # 'positive_mlp_loss': positive_mlp_loss.item(),
            # 'negative_mlp_loss': negative_mlp_loss.item(),
            'loss': loss.item()
        }

        # log = {
        #     **regularization_log,
        #     'positive_sample_loss': loss_log['pos_loss'],
        #     'negative_sample_loss': loss_log['neg_loss'],
        #     'loss': loss_log['loss']
        # }

        return log

    @staticmethod
    def test_step(model, g, train_triples, valid_triples, test_triples, args, mode="valid", random_sampling=True):
        '''
        Evaluate the model on test or valid datasets
        '''
        eval_dataset = TestDataset(g, train_triples, valid_triples, test_triples)

        sampler_head = eval_dataset.create_sampler(mode, args.batch_size,
                                                         args.negative_sample_size_eval,
                                                         args.negative_sample_size_eval,
                                                         args.eval_filter,
                                                         mode="chunk-head",
                                                         num_workers=args.num_workers,
                                                         rank=0, ranks=1)
        sampler_tail = eval_dataset.create_sampler(mode, args.batch_size,
                                                         args.negative_sample_size_eval,
                                                         args.negative_sample_size_eval,
                                                         args.eval_filter,
                                                         mode='chunk-tail',
                                                         num_workers=args.num_workers,
                                                         rank=0, ranks=1)

        model.eval()

        test_dataset_list = [sampler_head, sampler_tail]

        step = 0

        logs = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for pos_g, neg_g in tqdm(test_dataset):
                    mode = "tail-batch-test"
                    if neg_g.neg_head:
                        mode = "head-batch-test"
                    try:
                        positive_score, _ = model.forward(pos_g, mode="single", gpu_id=-1, trace=False, neg_deg_sample=False)
                        negative_score, _ = model.forward([pos_g, neg_g], mode=mode, gpu_id=-1, trace=False,
                                                   neg_deg_sample=False)
                    except:
                        print("relation: ", pos_g.edata["emb"].size())
                        print("entity", pos_g.ndata["emb"].size())
                        continue
                    batch_size = positive_score.size(0)
                    for i in range(batch_size):
                        ranking = torch.sum(negative_score > positive_score[i], dim=0)
                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    # if step % args.test_log_steps == 0:
                    #     logging.info('Evaluating the model... (%d)' % (step))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics

    @staticmethod
    def test_down_stream(model):
        downstream_test_dataset = DownStreamTaskTest()
        metrics = dict()
        for task, headIds, relaIds, tailIds, labels in downstream_test_dataset.generate_downstream_dataset():
            # logging.info("task: {}".format(task))
            headIds, relaIds, tailIds, labels = torch.tensor(headIds), torch.tensor(relaIds), torch.tensor(tailIds), torch.tensor(labels)
            preds = model.forward_test(headIds, relaIds, tailIds)
            roc = roc_auc(labels.detach().cpu().numpy(), preds[:, 1].detach().cpu().numpy())
            pr = pr_auc(labels.detach().cpu().numpy(), preds[:, 1].detach().cpu().numpy())
            fold = task.split("-")[-1]
            task_key = task.replace(fold, "")
            if task_key not in metrics:
                metrics[task_key] = dict()
                metrics[task_key]["roc"] = [roc]
                metrics[task_key]['pr'] = [pr]
                metrics[task_key]['fold'] = [fold]
            else:
                metrics[task_key]["roc"].append(roc)
                metrics[task_key]["pr"].append(pr)
                metrics[task_key]["fold"].append(fold)

        for task_key, roc_pr in metrics.items():
            print("task: {}, roc: ".format(task_key))
            cacMeanStd(roc_pr["roc"])
            print("task: {}, pr: ".format(task_key))
            cacMeanStd(roc_pr["pr"])
            for idx, fold in enumerate(roc_pr["fold"]):
                print("fold: {}, roc: {}, pr: {}".format(fold, roc_pr["roc"][idx], roc_pr["pr"][idx]))
            print("="*100)


class SparseEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    args :
        Global configs.
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """
    def __init__(self, args, num, dim, device):
        self.gpu = "0"
        self.args = args
        self.num = num
        self.trace = []

        self.emb = torch.empty(num, dim, dtype=torch.float32, device=device)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0
        self.has_cross_rel = False
        # queue used by asynchronous update
        self.async_q = None
        # asynchronous update process
        self.async_p = None

    def init(self, emb_init):
        """Initializing the embeddings.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        INIT.uniform_(self.emb, -emb_init, emb_init)
        INIT.zeros_(self.state_sum)

    def setup_cross_rels(self, cross_rels, global_emb):
        cpu_bitmap = torch.zeros((self.num,), dtype=torch.bool)
        for i, rel in enumerate(cross_rels):
            cpu_bitmap[rel] = 1
        self.cpu_bitmap = cpu_bitmap
        self.has_cross_rel = True
        self.global_emb = global_emb

    def get_noncross_idx(self, idx):
        cpu_mask = self.cpu_bitmap[idx]
        gpu_mask = ~cpu_mask
        return idx[gpu_mask]

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
        self.emb.share_memory_()
        self.state_sum.share_memory_()

    def __call__(self, idx, gpu_id=-1, trace=True):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        if self.has_cross_rel:
            cpu_idx = idx.cpu()
            cpu_mask = self.cpu_bitmap[cpu_idx]
            cpu_idx = cpu_idx[cpu_mask]
            cpu_idx = torch.unique(cpu_idx)
            if cpu_idx.shape[0] != 0:
                cpu_emb = self.global_emb.emb[cpu_idx]
                self.emb[cpu_idx] = cpu_emb.cuda(gpu_id)
        s = self.emb[idx]
        if gpu_id >= 0:
            s = s.cuda(gpu_id)
        # During the training, we need to trace the computation.
        # In this case, we need to record the computation path and compute the gradients.
        if trace:
            data = s.clone().detach().requires_grad_(True)
            self.trace.append((idx, data))
        else:
            data = s
        return data

    def update(self, gpu_id=-1):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        self.state_step += 1
        with torch.no_grad():
            for idx, data in self.trace:
                grad = data.grad.data

                clr = self.args.learning_rate
                # clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])
                # clr = self.args.learning_rate / (1 + (self.state_step - 1) * 0.98)

                # the update is non-linear so indices must be unique
                grad_indices = idx
                grad_values = grad

                grad_sum = (grad_values * grad_values).mean(1)
                device = self.state_sum.device
                if device != grad_indices.device:
                    grad_indices = grad_indices.to(device)
                if device != grad_sum.device:
                    grad_sum = grad_sum.to(device)

                self.state_sum.index_add_(0, grad_indices, grad_sum)
                std = self.state_sum[grad_indices]  # _sparse_mask
                if gpu_id >= 0:
                    std = std.cuda(gpu_id)
                std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                tmp = (-clr * grad_values / std_values)
                # tmp = -clr * grad_values
                if tmp.device != device:
                    tmp = tmp.to(device)
                # TODO(zhengda) the overhead is here.
                self.emb.index_add_(0, grad_indices, tmp)
        self.trace = []

    def finish_async_update(self):
        """Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None))
        self.async_p.join()

    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data in self.trace]
        return torch.cat(data, 0)

    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        self.emb = torch.Tensor(np.load(file_name))
