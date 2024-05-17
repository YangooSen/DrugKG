#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 17:45
# @Author  : yulong
# @File    : run.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import json
import torch

import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from dataloader import *
from model import *

from downstreamcode import *

import logging

logging.getLogger().setLevel(logging.INFO)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_downstream', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--has_edge_importance', action='store_true',
                        help='Allow providing edge importance score for each edge during training.' \
                             'The positive score will be adjusted ' \
                             'as pos_score = pos_score * edge_importance')
    parser.add_argument('--eval_filter', action='store_true',
                        help='Disable filter positive edges from randomly constructed negative edges for evaluation')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default='data/CarbonSiliconKG')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-te', '--triple_entity_embedding', action='store_true')
    parser.add_argument('-tr', '--triple_relation_embedding', action='store_true')
    parser.add_argument('--negative_sample_size_eval', type=int, default=500,
                        help='number of negative samples when evaluating training triples')
    parser.add_argument('-n', '--negative_sample_size', default=500, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=500, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-mlp_lr', '--mlp_learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=1, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=200, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    parser.add_argument('--save_checkpoint_steps', default=50000, type=int)
    parser.add_argument('--valid_steps', default=50000, type=int)
    parser.add_argument('--downstream_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--entity_emb_name',type=str,default=None)
    
    parser.add_argument('--train_triples',type=str,default=None)
    parser.add_argument('--valid_triples',type=str,default=None)
    parser.add_argument('--test_triples',type=str,default=None)
    
    
    parser.add_argument('--entities_fix',type=str,default=None)
    parser.add_argument('--relation_fix',type=str,default=None)
    parser.add_argument('--relation_emb_name',type=str,default=None)
    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.emb.cpu().numpy()
    np.save(
        os.path.join(args.save_path, args.entity_emb_name),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.emb.cpu().numpy()
    np.save(
        os.path.join(args.save_path, args.relation_emb_name),
        relation_embedding
    )


def read_triple(file_path, entity2id, relation2id, triples=dict()):
    '''
    Read triples and map them into ids.
    '''
    with open(file_path) as fin:
        head, relation, tail = list(), list(), list()
        for line in fin:
            if not line.strip():
                continue
            h, r, t = line.strip().split('\t')
            if h not in entity2id or t not in entity2id:
                continue
            head.append(entity2id[h])
            relation.append(relation2id[r])
            tail.append(entity2id[t])
    triples["head"] = head
    triples["relation"] = relation
    triples["tail"] = tail
    print('Finished. Read {} triples.'.format(len(triples["head"])))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    args.cuda = True
    args.do_train = True
    args.do_valid = True
    args.do_downstream = False
    args.do_test = True
    args.data_path = "/home/yangsen/pingce/kgdata/phkg/"
    # args.model = "pairRE"
    args.negative_sample_size = 500
    args.batch_size = 16 
    args.negative_sample_size_eval = 500
    args.test_batch_size = 16
    args.hidden_dim = 512
    args.gamma = 6
    args.adversarial_temperature = 1.0
    args.negative_adversarial_sampling = True
    args.learning_rate = 0.01
    args.mlp_learning_rate = 0.01
    args.max_steps = 200000
    # args.save_path = "model/codata_512/"
    args.regularization = 0.000000001
    args.double_entity_embedding = True
    args.double_relation_embedding = True

#    args.triple_relation_embedding = True 
#    args.triple_entity_embedding = False
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    with open(os.path.join(args.data_path, args.entities_fix)) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, args.relation_fix)) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation


    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, args.train_triples), entity2id, relation2id)
    # train_triples = read_triple(os.path.join(args.data_path, "umls_disease_triples_v2.txt"), entity2id, relation2id, train_triples)
    # train_triples = read_triple(os.path.join(args.data_path, "go_triples_fix.txt"), entity2id, relation2id, train_triples)
    # train_triples = read_triple(os.path.join(args.data_path, "ppi_triples.txt"), entity2id, relation2id, train_triples)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, args.valid_triples), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, args.test_triples), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = ConstructGraph(train_triples, valid_triples, test_triples, args)

    if args.do_train:
        train_dataset = TrainDataset(g, train_triples, args)

        train_sampler_head = train_dataset.create_sampler(args.batch_size,
                                                       args.negative_sample_size,
                                                       args.negative_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler_tail = train_dataset.create_sampler(args.batch_size,
                                                       args.negative_sample_size,
                                                       args.negative_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)

        train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                        args.negative_sample_size, args.negative_sample_size,
                                                        True, args.nentity)

    train_data = None
    gc.collect()

    kge_model = KGEModel(
        args=args,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        device=device,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        triple_relation_embedding=args.triple_relation_embedding,
        triple_entity_embedding=args.triple_entity_embedding,
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    current_learning_rate = args.mlp_learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()),
        lr=current_learning_rate
    )

    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):
            log = kge_model.train_step(kge_model, optimizer, train_sampler, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                args.learning_rate = args.learning_rate / 10
                # args.learning_rate = args.learning_rate
                current_learning_rate = args.learning_rate
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=args.mlp_learning_rate / 10
                )
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0 and step != init_step:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_downstream and step % args.downstream_steps == 0:
                logging.info('Evaluating on downstream Dataset...')
                kge_model.test_down_stream(kge_model)

            if args.do_valid and step % args.valid_steps == 0 and step != init_step:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, g, train_triples, valid_triples, test_triples, args, "valid")
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, save_variable_list, args)


    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, g, train_triples, valid_triples, test_triples, args, "valid")
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, g, train_triples, valid_triples, test_triples, args, "test")
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())

#example cmd
"""
python phkg.py --model DistMult -d 512 --train_triples train_0.txt --test_triples test_0.txt --valid_triples valid_0.txt --entities_fix entities_fix_0.dict
--relation_fix relations_fix1.dict -save kgmodel/phkg_DistMult/repo_0/ --entity_emb_name entity_embedding_0 --relation_emb_name relation_embedding_0
"""

