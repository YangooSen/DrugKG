#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 14:07
# @Author  : yulong
# @File    : kg_align.py

import argparse
import hashlib
import os


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, default='data/csKG')

    return parser.parse_args(args)


class KGAlignment(object):
    def __init__(self, args):
        super(KGAlignment, self).__init__()
        self.entity2id, self.relation2id, self.drug_map = dict(), dict(), dict()
        with open(os.path.join(args.data_path, 'entities_fix1.dict')) as fin:
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity2id[entity] = int(eid)
        with open(os.path.join(args.data_path, 'relations_fix1.dict')) as fin:
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation2id[relation] = int(rid)
        with open(os.path.join(args.data_path, 'drugs-node-inchikey.csv'))as fin:
            for line in fin:
                drug_id, smiles, wash_smiles, inchikey = line.strip().split(",")
                if inchikey == "None":
                    inchikey = drug_id
                self.drug_map[drug_id] = inchikey

    def align_drug(self, drug_id):
        if drug_id in self.entity2id:
            return self.entity2id[drug_id]
        elif self.drug_map[drug_id] in self.entity2id:
            return self.entity2id[self.drug_map[drug_id]]
        elif self.get_md5(self.drug_map[drug_id]) in self.entity2id:
            return self.entity2id[self.get_md5(self.drug_map[drug_id])]
        else:
            raise RuntimeError("invalid drug_id: {}".format(drug_id))

    def align_target(self, uniprot_id):
        if uniprot_id in self.entity2id:
            return self.entity2id[uniprot_id]
        elif self.get_md5(uniprot_id) in self.entity2id:
            return self.entity2id[self.get_md5(uniprot_id)]
        else:
            raise RuntimeError("invalid uniprot id: {}".format(uniprot_id))

    def align_disease(self, disease_id):
        if disease_id in self.entity2id:
            return self.entity2id[disease_id]
        elif self.get_md5(disease_id) in self.entity2id:
            return self.entity2id[self.get_md5(disease_id)]
        elif disease_id.replace("UMESH", "") in self.entity2id:
            return self.entity2id[disease_id.replace("UMESH", "")]
        elif self.get_md5(disease_id.replace("UMESH", "")) in self.entity2id:
            return self.entity2id[
                self.get_md5(disease_id.replace("UMESH", ""))
            ]

    def get_md5(self, key):
        md5obj = hashlib.md5()
        md5obj.update(key.encode("utf8"))
        hash = md5obj.hexdigest()
        return hash


if __name__ == '__main__':
    drug_id = ""
    disease_id = ""
    kga = KGAlignment(parse_args())
    kga.align_drug(drug_id)
    kga.align_disease(disease_id)
