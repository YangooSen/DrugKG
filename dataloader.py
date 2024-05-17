#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 18:02
# @Author  : yulong
# @File    : dataloader.py
import os.path
from tqdm import tqdm
import dgl
import numpy as np
import scipy as sp
import dgl.backend as F
import torch
import logging
import gc
import random
from dgl.base import NID, EID
from utils.utils import *
import collections
from torch.utils.data import Dataset, DataLoader

logging.getLogger().setLevel(logging.INFO)

entities_dict = "entities_fix1.dict"
relations_dict = "relations_fix1.dict"
train_txt = "train_fix1.txt"
valid_txt = "valid_fix1.txt"
test_txt = "test_fix1.txt"


def ConstructGraph(train_triples, valid_triples, test_triples, args):
    src, etype_id, dst = list(), list(), list()
    for triples in [train_triples, valid_triples, test_triples]:
        src.append(np.array(triples['head'], dtype=np.int64))
        etype_id.append(np.array(triples['relation'], dtype=np.int64))
        dst.append(np.array(triples['tail'], dtype=np.int64))

    src = np.concatenate(src)
    etype_id = np.concatenate(etype_id)
    dst = np.concatenate(dst)

    n_entities = args.nentity

    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])

    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)

    g.edata['tid'] = F.tensor(etype_id, F.int64)

    return g


class ChunkNegEdgeSubgraph(dgl.DGLGraph):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid


def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)


class EvalSampler(object):
    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g,
                                        self.neg_chunk_size,
                                        self.neg_sample_size,
                                        'chunk' in self.mode,
                                        self.neg_head,
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        self.sampler_iter = iter(self.sampler)
        return self


class TrainDataset(object):
    def __init__(self, g, train_triples, args):
        num_train = len(train_triples["head"])
        print("|Train|:", num_train)

        self.edge_parts = [np.arange(num_train)]
        self.rel_parts = [np.arange(args.nrelation)]

        self.g = g

        self.args = args

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')

        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank], dtype=F.int64),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode = mode,
                           num_workers=num_workers,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)


class TestDataset(object):
    def __init__(self, g, train_triples, valid_triples, test_triples):
        self.num_train = len(train_triples["head"])
        self.num_test = len(test_triples["head"])
        self.num_valid = len(valid_triples["head"])
        print("|valid|:", self.num_valid)
        print("|test|:", self.num_test)

        self.g = g

        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples

        self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())

    def get_edges(self, eval_type):
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        if eval_type == 'valid':
            return self.valid_triples
        elif eval_type == 'test':
            return self.test_triples
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)


class NewBidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes):
        while True:
             for pos_g, neg_g in dataloader:
                 neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                             is_chunked, neg_head, num_nodes)
                 if neg_g is None:
                     continue

                 pos_g.ndata['id'] = pos_g.parent_nid
                 neg_g.ndata['id'] = neg_g.parent_nid
                 pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                 yield pos_g, neg_g


class DownStreamTaskTrain(object):
    def __init__(self, KGVersion="csKG"):
        self.taget_node_id = "data/downstream/disease-target/target-node-id.txt"
        self.disease_target_alzheimer = "data/downstream/disease-target/alzheimer"
        self.disease_target_breastCancer = "data/downstream/disease-target/breastCancer"
        self.disease_target_colorectalCancer = "data/downstream/disease-target/colorectalCancer"
        self.disease_target_lungCancer = "data/downstream/disease-target/lungCancer"
        self.disease_target_prostateCancer = "data/downstream/disease-target/prostateCancer"

        self.drug_node_id_ = "data/downstream/drug-disease/drugs-node-inchikey.csv"
        self.drug_disease = "data/downstream/drug-disease/data01"

        self.drug_node_id = "data/downstream/drug-target/comp_struc_inchikey.csv"
        self.drug_target_drug_coldstart = "data/downstream/drug-target/drug_coldstart"
        self.drug_target_protein_coldstart = "data/downstream/drug-target/protein_coldstart"
        self.drug_target_warm_start_1_10 = "data/downstream/drug-target/warm_start_1_10"

        self.carbonSiliconKG = "data/{}".format(KGVersion)
        self.train_triple_path = "data/{}/{}".format(KGVersion, train_txt)

        self.triples = list()
        self.parse_disease_target()
        self.parse_drug_target()
        self.load_drug_disease()

        self.add_to_kg()

    def parse_disease_target(self):
        self.taget_node_id_mapping = dict()
        for line in iterFile(self.taget_node_id):
            self.taget_node_id_mapping[line.strip().split("\t")[1]] = getMd5(line.strip().split("\t")[0])

        dirs = [self.disease_target_alzheimer, self.disease_target_breastCancer, self.disease_target_colorectalCancer,
               self.disease_target_lungCancer, self.disease_target_prostateCancer]
        trains = ["alzheimerIntAct{}indicesofTrainSet.txt", "breastCancerIntAct{}indicesofTrainSet.txt",
                 "colorectalCancerIntAct{}indicesofTrainSet.txt", "lungCancerIntAct{}indicesofTrainSet.txt",
                 "prostateCancerIntAct{}indicesofTrainSet.txt"]
        labels = ["alzheimerIntAct{}trainLabel.txt", "breastCancerIntAct{}trainLabel.txt",
                 "colorectalCancerIntAct{}trainLabel.txt", "lungCancerIntAct{}trainLabel.txt",
                 "prostateCancerIntAct{}trainLabel.txt"]
        diseaseMd5 = ["71398073d74285987021fb7c4e1f0601", "881d8487bb2cdac60da65adbef5ede07",
                      "b36109c1040e4efdea056ddc16acb38e", "c8c9ed8724717ce959b9c1cb6c86b877",
                      "2ac620d4dd4195ad2ae9726ae7da65fd"]
        for idx, dir in enumerate(dirs):
            train = trains[idx]
            label = labels[idx]
            disMd5 = diseaseMd5[idx]
            self.load_disease_target(train, label, dir, disMd5)

    def load_disease_target(self, train, label, dir, disMd5):
        for fold in range(11, 16):
            train_fold_path = os.path.join(dir, train.format(fold))
            label_fold_path = os.path.join(dir, label.format(fold))
            target_node = [line.strip() for line in iterFile(train_fold_path) if line.strip()]
            labels = [line.strip() for line in iterFile(label_fold_path) if line.strip()]
            [self.triples.append("{}\ttarget-disease\t{}".format(getMd5(self.taget_node_id_mapping[target_node[idx]]), disMd5))
             for idx, label in enumerate(labels) if label == "1"]

    def parse_drug_target(self):
        self.drug_node_id_mapping = dict()
        for line in iterFile(self.drug_node_id):
            if line.startswith("head"):
                continue
            drug_id, drug_inchikey = line.strip().split(",")[0], line.strip().split(",")[4]
            if drug_inchikey == "None":
                drug_inchikey = drug_id
            else:
                drug_inchikey = getMd5(drug_inchikey)
            self.drug_node_id_mapping[drug_id] = drug_inchikey

        dirs = [self.drug_target_drug_coldstart, self.drug_target_protein_coldstart, self.drug_target_warm_start_1_10]
        train = "train_fold_{}.csv"
        for idx, dir in enumerate(dirs):
            self.load_drug_target(train, dir)

    def load_drug_target(self, train, dir):
        for fold in range(1, 11):
            train_fold_path = os.path.join(dir, train.format(fold))
            [self.triples.append("{}\tdrug-target\t{}".format(self.drug_node_id_mapping[line.strip().split(",")[0]], getMd5(line.strip().split(",")[2])))
             for line in iterFile(train_fold_path) if
             not line.startswith("head") and line.strip().split(",")[3] == "1.0"]

    def load_drug_disease(self):
        self.drug_node_id_mapping = dict()
        for line in iterFile(self.drug_node_id_):
            drug_id, drug_inchikey = line.strip().split(",")[0], line.strip().split(",")[3]
            if drug_inchikey == "None":
                drug_inchikey = drug_id
            else:
                drug_inchikey = getMd5(drug_inchikey)
            self.drug_node_id_mapping[drug_id] = drug_inchikey

        for fold in range(0, 5):
            train_fold_path = os.path.join(self.drug_disease, "train_df_{}.csv".format(fold))
            [self.triples.append("{}\tdrug-disease\t{}".format(self.drug_node_id_mapping[line.strip().split(",")[0]],
                                                               getMd5(line.strip().split(",")[1].split(":")[1]))) for
             line in iterFile(train_fold_path) if
             not line.startswith("drug") and line.strip().split(",")[2] == "1" and line.strip().split(",")[
                 0] in self.drug_node_id_mapping]

    def add_to_kg(self):
        with open(os.path.join(self.carbonSiliconKG, entities_dict)) as fin:
            self.entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity2id[entity] = int(eid)

        with open(os.path.join(self.carbonSiliconKG, relations_dict)) as fin:
            self.relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation2id[relation] = int(rid)

        relation_add, entity_add = list(), list()
        relations = ["target-disease", "drug-target", "drug-disease"]
        for rela in relations:
            rid = len(self.relation2id.keys())
            if rela not in self.relation2id:
                self.relation2id[rela] = rid
                relation_add.append("{}\t{}\n".format(rid, rela))

        fp = open(self.train_triple_path, "a", encoding="utf8")
        for triple in self.triples:
            head, rela, tail = triple.split("\t")
            eid = len(self.entity2id.keys())
            if head not in self.entity2id:
                self.entity2id[head] = eid
                entity_add.append("{}\t{}\n".format(eid, head))
            if tail not in self.entity2id:
                self.entity2id[tail] = eid + 1
                entity_add.append("{}\t{}\n".format(eid, tail))
            fp.write(triple + "\n")
        fp.close()

        with open(os.path.join(self.carbonSiliconKG, entities_dict), "a", encoding="utf8")as f:
            for ent in entity_add:
                f.write(ent)

        with open(os.path.join(self.carbonSiliconKG, relations_dict), "a", encoding="utf8") as f:
            for rela in relation_add:
                f.write(rela)


class DownStreamTaskTest(Dataset):
    def __init__(self, KGVersion="csKG"):
        self.taget_node_id = "data/downstream/disease-target/target-node-id.txt"
        self.disease_target_alzheimer = "data/downstream/disease-target/alzheimer"
        self.disease_target_breastCancer = "data/downstream/disease-target/breastCancer"
        self.disease_target_colorectalCancer = "data/downstream/disease-target/colorectalCancer"
        self.disease_target_lungCancer = "data/downstream/disease-target/lungCancer"
        self.disease_target_prostateCancer = "data/downstream/disease-target/prostateCancer"

        self.drug_node_id_ = "data/downstream/drug-disease/drugs-node-inchikey.csv"
        self.drug_disease = "data/downstream/drug-disease/data01"

        self.drug_node_id = "data/downstream/drug-target/comp_struc_inchikey.csv"
        self.drug_target_drug_coldstart = "data/downstream/drug-target/drug_coldstart"
        self.drug_target_protein_coldstart = "data/downstream/drug-target/protein_coldstart"
        self.drug_target_warm_start_1_10 = "data/downstream/drug-target/warm_start_1_10"

        self.carbonSiliconKG = "data/{}".format(KGVersion)
        self.carbonSiliconKG_train = "data/{}/{}".format(KGVersion, train_txt)
        self.carbonSiliconKG_train_filter = "data/{}/train_filter_leakage.txt".format(KGVersion)

        self.triples = dict()
        self.triples_mismatch = dict()
        self.load_kg()

        self.parse_disease_target()
        self.parse_drug_target()

        self.load_drug_disease()

    def load_kg(self):
        with open(os.path.join(self.carbonSiliconKG, entities_dict)) as fin:
            self.entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity2id[entity] = int(eid)

        with open(os.path.join(self.carbonSiliconKG, relations_dict)) as fin:
            self.relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation2id[relation] = int(rid)

    def parse_disease_target(self):
        self.taget_node_id_mapping = dict()
        for line in iterFile(self.taget_node_id):
            self.taget_node_id_mapping[line.strip().split("\t")[1]] = line.strip().split("\t")[0]

        dirs = [self.disease_target_alzheimer, self.disease_target_breastCancer, self.disease_target_colorectalCancer,
               self.disease_target_lungCancer, self.disease_target_prostateCancer]
        trains = ["alzheimerIntAct{}indicesofTestSet.txt", "breastCancerIntAct{}indicesofTestSet.txt",
                 "colorectalCancerIntAct{}indicesofTestSet.txt", "lungCancerIntAct{}indicesofTestSet.txt",
                 "prostateCancerIntAct{}indicesofTestSet.txt"]
        labels = ["alzheimerIntAct{}testLabel.txt", "breastCancerIntAct{}testLabel.txt",
                 "colorectalCancerIntAct{}testLabel.txt", "lungCancerIntAct{}testLabel.txt",
                 "prostateCancerIntAct{}testLabel.txt"]
        diseaseMd5 = ["71398073d74285987021fb7c4e1f0601", "881d8487bb2cdac60da65adbef5ede07",
                      "b36109c1040e4efdea056ddc16acb38e", "c8c9ed8724717ce959b9c1cb6c86b877",
                      "2ac620d4dd4195ad2ae9726ae7da65fd"]
        for idx, dir in enumerate(dirs):
            tag = dir.split("/")[-1]
            train = trains[idx]
            label = labels[idx]
            disMd5 = diseaseMd5[idx]
            self.load_disease_target(train, label, dir, disMd5, tag)

    def load_disease_target(self, train, label, dir, disMd5, tag):
        for fold in range(11, 16):
            train_fold_path = os.path.join(dir, train.format(fold))
            label_fold_path = os.path.join(dir, label.format(fold))
            target_node = [line.strip() for line in iterFile(train_fold_path) if line.strip()]
            labels = [line.strip() for line in iterFile(label_fold_path) if line.strip()]
            tmp, tmp_mismatch = list(), list()
            for idx, lab in enumerate(labels):
                mismatch = False
                if getMd5(self.taget_node_id_mapping[target_node[idx]]) in self.entity2id:
                    head = self.entity2id[getMd5(self.taget_node_id_mapping[target_node[idx]])]
                elif self.taget_node_id_mapping[target_node[idx]] in self.entity2id:
                    head = self.entity2id[self.taget_node_id_mapping[target_node[idx]]]
                else:
                    mismatch, head = True, self.taget_node_id_mapping[target_node[idx]]

                relation, tail = self.relation2id["target-disease"], self.entity2id[disMd5]

                if mismatch is True:
                    tmp_mismatch.append("{}\ttarget-disease\t{}\t{}".format(head, disMd5, lab))
                    continue

                tmp.append("{}\t{}\t{}\t{}".format(head, relation, tail, lab))

            self.triples[tag+"-"+str(fold)] = tmp
            self.triples_mismatch[tag+"-"+str(fold)] = tmp_mismatch

    def parse_drug_target(self):
        self.drug_node_id_mapping = dict()
        for line in iterFile(self.drug_node_id):
            if line.startswith("head"):
                continue
            drug_id, drug_inchikey = line.strip().split(",")[0], line.strip().split(",")[4]
            if drug_inchikey == "None":
                drug_inchikey = drug_id
            self.drug_node_id_mapping[drug_id] = drug_inchikey

        dirs = [self.drug_target_drug_coldstart, self.drug_target_protein_coldstart, self.drug_target_warm_start_1_10]
        train = "test_fold_{}.csv"
        for idx, dir in enumerate(dirs):
            tag = dir.split("/")[-1]
            self.load_drug_target(train, dir, tag)

    def load_drug_target(self, train, dir, tag):
        for fold in range(1, 11):
            train_fold_path = os.path.join(dir, train.format(fold))
            tmp, tmp_mismatch = list(), list()

            for line in iterFile(train_fold_path):
                if line.startswith("head"):
                    continue

                mismatch = False

                drug, relation, target, label = line.strip().split(",")

                if target in self.entity2id:
                    targets = self.entity2id[target]
                elif getMd5(target) in self.entity2id:
                    targets = self.entity2id[getMd5(target)]
                else:
                    mismatch, targets = True, target

                if drug in self.entity2id:
                    drugs = self.entity2id[drug]
                elif getMd5(drug) in self.entity2id:
                    drugs = self.entity2id[getMd5(drug)]
                elif self.drug_node_id_mapping[drug] in self.entity2id:
                    drugs = self.entity2id[self.drug_node_id_mapping[drug]]
                elif getMd5(self.drug_node_id_mapping[drug]) in self.entity2id:
                    drugs = self.entity2id[getMd5(self.drug_node_id_mapping[drug])]
                else:
                    mismatch, drugs = True, drug

                relations = self.relation2id["drug-target"]
                label = int(label.split(".")[0])

                if mismatch is True:
                    tmp_mismatch.append("{}\t{}\t{}\t{}".format(drugs, relations, targets, label))
                    continue

                tmp.append("{}\t{}\t{}\t{}".format(drugs, relations, targets, label))

            self.triples[tag+"-"+str(fold)] = tmp
            self.triples_mismatch[tag+"-"+str(fold)] = tmp_mismatch

    def load_drug_disease(self):
        self.drug_node_id_mapping = dict()
        for line in iterFile(self.drug_node_id_):
            drug_id, drug_inchikey = line.strip().split(",")[0], line.strip().split(",")[3]
            if drug_inchikey == "None":
                drug_inchikey = drug_id
            self.drug_node_id_mapping[drug_id] = drug_inchikey

        for fold in range(0, 5):
            tmp, tmp_mismatch = list(), list()
            train_fold_path = os.path.join(self.drug_disease, "test_df_{}.csv".format(fold))

            for line in iterFile(train_fold_path):
                if line.startswith("drug"):
                    continue
                mismatch = False
                drug, disease, label, SMILES = line.strip().split(",")

                if drug in self.entity2id:
                    drugs = self.entity2id[drug]
                elif getMd5(drug) in self.entity2id:
                    drugs = self.entity2id[drug]
                elif drug not in self.drug_node_id_mapping:
                    mismatch, drugs = True, drug
                elif self.drug_node_id_mapping[drug] in self.entity2id:
                    drugs = self.entity2id[self.drug_node_id_mapping[drug]]
                elif getMd5(self.drug_node_id_mapping[drug]) in self.entity2id:
                    drugs = self.entity2id[getMd5(self.drug_node_id_mapping[drug])]
                else:
                    mismatch, drugs = True, drug

                if disease in self.entity2id:
                    diseases = self.entity2id[disease]
                elif getMd5(disease) in self.entity2id:
                    diseases = self.entity2id[getMd5(disease)]
                elif disease.split(":")[1] in self.entity2id:
                    diseases = self.entity2id[disease.split(":")[1]]
                elif getMd5(disease.split(":")[1]) in self.entity2id:
                    diseases = self.entity2id[getMd5(disease.split(":")[1])]
                else:
                    mismatch, diseases = True, disease

                relations = self.relation2id["drug-disease"]

                if mismatch is True:
                    tmp_mismatch.append("{}\t{}\t{}\t{}".format(drugs, relations, diseases, label))
                    continue

                tmp.append("{}\t{}\t{}\t{}".format(drugs, relations, diseases, label))

            self.triples["drug-disease-{}".format(fold)] = tmp
            self.triples_mismatch["drug-disease-{}".format(fold)] = tmp_mismatch

    def check_test_leakage(self):
        ''' 训练前使用：需要去除可能泄露的数据 '''
        self.test_triples = set()
        triple_list = list()
        [triple_list.extend(v) for k, v in self.triples.items()]
        [self.test_triples.add(("".format(int(triple.split("\t")[0])),
                                "".format(int(triple.split("\t")[2])))) for triple in triple_list if int(triple.strip().split("\t")[3]) == 1]
        self.train_triple_filter = list()
        for line in iterFile(self.carbonSiliconKG_train):
            if not line.strip() or len(line.strip().split("\t")) != 3:
                continue
            head, rela, tail = line.strip().split("\t")
            headIdx, tailIdx = self.entity2id[head], self.entity2id[tail]
            if (headIdx, tailIdx) not in self.test_triples and (tailIdx, headIdx) not in self.test_triples:
                self.train_triple_filter.append(line.strip())

        saveFile(self.carbonSiliconKG_train_filter, self.train_triple_filter)

    def add_test_node_kg(self):
        test_nodeset = list()
        triple_list = list()
        [triple_list.extend(v) for k, v in self.triples_mismatch.items()]
        for triple in triple_list:
            eid = len(self.entity2id.keys())
            head, rela, tail, label = triple.strip().split("\t")
            if not head.isdigit() and head not in self.entity2id:
                self.entity2id[head] = eid
                test_nodeset.append("{}\t{}\n".format(eid, head))

            if not tail.isdigit() and tail not in self.entity2id:
                self.entity2id[tail] = eid + 1
                test_nodeset.append("{}\t{}\n".format(eid, tail))

        with open(os.path.join(self.carbonSiliconKG, entities_dict), "a", encoding="utf8")as f:
            for ent in test_nodeset:
                f.write(ent)

    def generate_downstream_dataset(self):
        ''' 训练中使用：生成下游的dataloader '''
        for downstreamTag, triple_list in self.triples.items():
            headIds = [int(triple.strip().split("\t")[0]) for triple in triple_list]
            relaIds = [int(triple.strip().split("\t")[1]) for triple in triple_list]
            tailIds = [int(triple.strip().split("\t")[2]) for triple in triple_list]
            labels = [int(triple.strip().split("\t")[3]) for triple in triple_list]
            yield downstreamTag, headIds, relaIds, tailIds, labels


class downstreamTestDataset(Dataset):
    def __init__(self, triples):
        self.len = len(triples)
        self.triples = triples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, rela, tail, label = self.triples[idx].split("\t")
        return int(head), int(
            rela), int(tail), int(label)


def addUMLS():
    umls_disease_entities = [line.strip().split("\t")[1] for line in iterFile("data/csKG/umls_disease_entities.dict") if line.strip()]
    umls_disease_relations = [line.strip().split("\t")[1] for line in iterFile("data/csKG/umls_disease_relations.dict") if line.strip()]
    with open("data/csKG/entities_fix1.dict", "r", encoding="utf8") as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open('data/csKG/relations_fix1.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    for umls_dis_ent in umls_disease_entities:
        id = len(entity2id)
        if umls_dis_ent not in entity2id:
            entity2id[umls_dis_ent] = id

    for umls_dis_rela in umls_disease_relations:
        id = len(relation2id)
        if umls_dis_rela not in relation2id:
            relation2id[umls_dis_rela] = id

    with open("data/csKG/entities_fix1_umls_v2.dict", "w", encoding="utf8")as f:
        for k, v in entity2id.items():
            f.write("{}\t{}\n".format(v, k))

    with open("data/csKG/relations_fix1_umls_v2.dict", "w", encoding="utf8")as f:
        for k, v in relation2id.items():
            f.write("{}\t{}\n".format(v, k))


def addGO_addPPI_updateUMLS_entities():
    # process ppi
    # PPI_triples = list()
    # for line in iterFile("data/csKG/string_ppi.csv"):
    #     if line.startswith("protein") or line.strip():
    #         continue
    #     protein1, protein2, combined_score, pro1_uni, pro2_uni = line.strip().split(",")
    #     PPI_triples.append("{}\tPPI\t{}\n".format(getMd5(pro1_uni), getMd5(pro2_uni)))
    # with open("data/csKG/ppi_triples.txt", "w", encoding="utf8")as f:
    #     f.write("".join(PPI_triples))

    # origin entities dict
    with open("data/csKG/entities_fix1.dict", "r", encoding="utf8")as fin:
        origin_entity2id = collections.OrderedDict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            origin_entity2id[entity] = int(eid)

    # umls
    with open("data/csKG/umls_disease_entities_v2.dict", "r", encoding="utf8") as fin:
        umls_entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            umls_entity2id[entity] = int(eid)
    # ppi
    with open("data/csKG/ppi_entities.dict", "r", encoding="utf8") as fin:
        ppi_entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            ppi_entity2id[entity] = int(eid)
    # go
    with open("data/csKG/go_entities_fix.dict", "r", encoding="utf8") as fin:
        go_entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            go_entity2id[entity] = int(eid)

    for dic in [umls_entity2id, ppi_entity2id, go_entity2id]:
        for key in dic:
            idx = len(origin_entity2id.keys())
            if key not in origin_entity2id:
                origin_entity2id[key] = idx

    with open("data/csKG/bulk_entities_fix.dict", "w", encoding="utf8") as f:
        for k, v in origin_entity2id.items():
            f.write("{}\t{}\n".format(v, k))


def merge_triples():
    bulk_filter_leakage = list()
    for line in iterFile("data/csKG/train_filter_leakage.txt"):
        bulk_filter_leakage.append(line)

    for line in iterFile("data/csKG/go_triples_fix.txt"):
        bulk_filter_leakage.append(line)

    for line in iterFile("data/csKG/ppi_triples.txt"):
        bulk_filter_leakage.append(line)

    for line in iterFile("data/csKG/umls_disease_triples_v2.txt"):
        bulk_filter_leakage.append(line)

    with open("data/csKG/bulk_filter_leakage.txt", "w", encoding="utf8")as f:
        for line in bulk_filter_leakage:
            f.write(line)


def alignDiseaseTargets():
    with open("data/csKG/bulk_entities_fix.dict", "r", encoding="utf8")as fin:
        origin_entity2id = collections.OrderedDict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            origin_entity2id[entity] = int(eid)

    target_data = list()
    for line in iterFile("data/downstream/disease-target/target-node-id.txt"):
        if line.strip():
            target, target_id = line.strip().split("\t")
            targetMd5 = getMd5(target)
            if targetMd5 in origin_entity2id:
                targetMd5Indice = origin_entity2id[targetMd5]
                target_data.append("{}\t{}\t{}\t{}\n".format(target, target_id, targetMd5, targetMd5Indice))
                continue

            if target in origin_entity2id:
                targetIndice = origin_entity2id[target]
                target_data.append("{}\t{}\t{}\t{}\n".format(target, target_id, target, targetIndice))

    with open("data/downstream/mapping/matchProtein4DiseaseTarget", "w", encoding="utf8") as f:
        for data in target_data:
            f.write(data)


def checkCTDDiseaseTargetsLeakeage(KGVersion):
    checkSet = set()
    dataset = dataSet("data/downstream/ctd-disease-target-large-warmstart-v2/")
    for line in dataset.parseTriples():
        if not line.strip():
            continue
        head, tail, label = line.strip().split("\t")
        if label == "1":
            checkSet.add((head, tail))

    trainFilterLeakeage = list()
    for line in iterFile("data/{}/train_fix1.txt".format(KGVersion)):
        if not line.strip():
            continue
        head, rela, tail = line.strip().split("\t")
        if (head, tail) not in checkSet and (tail, head) not in checkSet:
            trainFilterLeakeage.append(line)

    with open("data/{}/train_filter_ctd_leakage_large_v2.txt".format(KGVersion), "w", encoding="utf8")as f:
        for line in trainFilterLeakeage:
            f.write(line)


def relationCounter():
    relaCount = dict()
    for line in tqdm(iterFile("data/csKG/train_filter_leakage.txt")):
        lineSplit = line.strip().split("\t")
        if len(lineSplit) != 3:
            continue
        rela = lineSplit[1]
        if rela not in relaCount:
            relaCount[rela] = 0
        relaCount[rela] += 1
    print(relaCount)


def splitGraph(KGVersion="csKG"):
    filePath = "data/{}/triples.txt".format(KGVersion)
    testPath = "data/{}/{}".format(KGVersion, test_txt)
    trainPath = "data/{}/{}".format(KGVersion, train_txt)
    validPath = "data/{}/{}".format(KGVersion, valid_txt)

    logging.info("start load triples ...")

    triples, train_triples = dict(), list()
    for line in tqdm(iterFile(filePath)):
        lineSplit = line.strip().split("\t")
        if len(lineSplit) != 3:
            continue
        _, relation, _ = lineSplit
        if relation not in triples:
            triples[relation] = dict()
            triples[relation][line.strip()] = ""
        else:
            triples[relation][line.strip()] = ""

    train_triples = None
    gc.collect()

    [print("{}: {} ".format(k, len(v))) for k, v in triples.items()]
    logging.info("load finished ... start split dataset ...")

    # test, valid split
    for k, v in triples.items():
        fullTriplesNum, testData, validData = len(v.keys()), list(), list()
        sampleNumber = int(fullTriplesNum * 0.01)
        testData = random.sample(v.keys(), sampleNumber)
        [v.pop(test) for test in testData]
        validData = random.sample(v.keys(), sampleNumber)
        [v.pop(valid) for valid in validData]
        with open(testPath, "a", encoding="utf8") as f:
            f.write("\n".join(testData) + "\n")
        with open(validPath, "a", encoding="utf8") as f:
            f.write("\n".join(validData) + "\n")
        with open(trainPath, "a", encoding="utf8") as f:
            for item in list(v.keys()):
                f.write(item+"\n")

        logging.info("{} split finished ...".format(k))

    logging.info("split success ...")


def kgPreprocess():
    KGVersion = "csKG"
    # splitGraph(KGVersion)
    # logging.info("downstream tasks data add to train")
    # downstream_train = DownStreamTaskTrain(KGVersion) # 下游任务训练集数据对齐至KG中
    # logging.info("downstream tasks data testset check for leakeage")
    # downstream_test = DownStreamTaskTest(KGVersion) # 下游任务测试集数据需要做泄露排除
    # logging.info("check leakage ...")
    # downstream_test.check_test_leakage()
    # logging.info("add test node to kg")
    # downstream_test.add_test_node_kg()
    # logging.info("start ctd dataset ...")
    checkCTDDiseaseTargetsLeakeage(KGVersion)


if __name__ == '__main__':
    kgPreprocess()
    # addUMLS()
    # addGO_addPPI_updateUMLS_entities()
    # merge_triples()
    # alignDiseaseTargets()
    # relationCounter()
