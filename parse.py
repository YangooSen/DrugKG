from os import P_ALL, write
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

cnt=231

class counter():
    def __init__(self,path) -> None:
        
        _,fileName=os.path.split(path)
        self.name=fileName.split(".")[0]
        
        print("-"*15 +self.name)
        self.df=self.getDF(path)
        self.getData()
        #self.getRelationGraph() 
        print("-"*15)


    def getDF(self,path):
        df=pd.read_csv(path)
        if "primekg" not in path:
            df.columns=["head","relation","tail"]
        
        if 'iBKH' in path:
            df.fillna(value='Drug-Drug',inplace=True)

        elif 'hetionet' in path:
            df.fillna(value='Gene-Gene',inplace=True)
        
        return df

    def getUniqueEntity(self):
        uniqueEntity=np.unique(list(self.df['head'])+list(self.df['tail']))

        return uniqueEntity


    def getUniqueReltion(self):
        uniqueReltion=np.unique(list(self.df['relation']))
        if "nan" in uniqueReltion:
            print("there is nan")
            self.printNAN()


        return uniqueReltion

    def getRelationGraph(self):
        #count relation num
        cntDic={}
        for i,row in self.df.iterrows():
            cntDic[row['relation']]=cntDic.get(row['relation'],0)+1
        
        
        plt.subplot(cnt) 
        plt.bar(list(cntDic.keys()),list(cntDic.values()))        
        plt.xticks(rotation=270)
        plt.xlabel("relation")
        plt.ylabel("number")
        plt.title(self.name)
        #plt.tight_layout()
        #plt.savefig("./"+self.name+".png")


    def getData(self):
        with open("./data.txt","a") as f:
            self.print("data shape:{}".format(self.df.shape),f)
            self.uniqueEntity=self.getUniqueEntity()
            self.print("num of unique entity:{}".format(len(self.uniqueEntity)),f)
            self.uniqueRelation=self.getUniqueReltion()
            self.print("num of unique relation:{}".format(len(self.uniqueRelation)),f)
            self.print(self.uniqueRelation,f)


    def print(self,data,f):
        
        print("="*10)
        f.write("="*10)
        f.write("\n")


        print(data)
        f.write(str(data))
        f.write("\n")


        print("="*10)
        f.write("="*10)
        f.write("\n")




    def printNAN(self):
        print(self.df[self.df['relation'].isna()])        

datasets=[
        "./biokg_mapped.csv",
        "./drkg_mapped.csv",
        "./hetionet_kg_mapped.csv",
        "./iBKH_mapped.csv",
        "./primekg_mapped.csv",
        ]



fig=plt.figure(figsize=(12,15))

for dataset in datasets:
    c=counter(dataset)
    cnt+=1

fig.tight_layout()
plt.savefig("./res.png")
