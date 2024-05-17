import os
import hashlib
from sklearn import metrics


class dataSet(object):
    def __init__(self, fileDir):
        self.fileDirList = list()
        self.fileDir = fileDir
        self.fileDirList.append(fileDir)

    def addDir(self, fileDir):
        self.fileDirList.append(fileDir)

    def parseTriples(self):
        for file in os.listdir(self.fileDir):
            filePath = os.path.join(self.fileDir, file)
            for line in iterFile(filePath):
                yield line

    def parseTriplesBulk(self):
        for fileDir in self.fileDirList:
            for file in os.listdir(fileDir):
                filePath = os.path.join(fileDir, file)
                for line in iterFile(filePath):
                    yield line

    def parseTriplesConditional(self, condition, mode="distribute"):
        for file in os.listdir(self.fileDir):
            if mode == "single":
                if file in condition:
                    filePath = os.path.join(self.fileDir, file)
                    for line in iterFile(filePath):
                        yield line
            elif mode == "distribute":
                fileNum = file.split("-")[1:]
                if len(fileNum) == 1 and fileNum[0] in condition:
                    filePath = os.path.join(self.fileDir, file)
                    for line in iterFile(filePath):
                        yield line
                elif len(fileNum) == 2 and "-".join(fileNum) in condition:
                    filePath = os.path.join(self.fileDir, file)
                    for line in iterFile(filePath):
                        yield line


def roc_auc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def iterFile(file):
    with open(file, "r", encoding="utf8")as f:
        for line in f:
            yield line


def saveFile(file, data):
    with open(file, "w", encoding="utf8")as f:
        for item in data:
            f.write(item+"\n")


def getMd5(key):
    md5obj = hashlib.md5()
    md5obj.update(key.encode("utf8"))
    hash = md5obj.hexdigest()
    return hash


def cacMeanStd(key):
    import torch
    key = torch.tensor(key)
    mean = torch.mean(key)
    std = torch.std(key)
    print("mead: {}, std: {}".format(mean, std))


def tmpforDisTar():
    with open("../data/downstream/mapping/matchDisease4DiseaseTarget", "r", encoding="utf8")as f:
        md5Idlist = [line.strip().split("\t")[1] for line in f if line.strip()]

    entity2id = {}
    with open("../data/csKG/entities_fix1.dict", "r", encoding="utf8")as f:
        for line in f:
            if line.strip():
                id, md5Id = line.strip().split("\t")
                entity2id[md5Id] = int(id)

    for md5Id in md5Idlist:
        if md5Id in entity2id:
            print(md5Id, entity2id[md5Id])


if __name__ == '__main__':
    # roc = [0.7593316435910396, 0.832369315802776, 0.8541708808505984, 0.8589085540834337, 0.7950120061130788, 0.8393570034567744, 0.8393619924949379, 0.7749696449864997, 0.8430912768301283, 0.8438441991168205]
    # pr = [0.17578038106146676, 0.23750887998832182, 0.337467934119938, 0.25763477493293657, 0.22443878020122782, 0.2849326805220513, 0.26851033123941187, 0.19239759294552142, 0.28826006609525456, 0.24793512282195135]
    # cacMeanStd(roc)
    # cacMeanStd(pr)
    tmpforDisTar()
