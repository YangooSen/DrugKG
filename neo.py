from ctypes.wintypes import PRECT
import pandas as pd
from py2neo import Graph,Node,Relationship,NodeMatcher
import numpy as np



def printCols(df):
    data=dict(df.iloc[:5,:])
    for k,v in data.items():
        print(k)
        print(v)
        print("="*10)


# 20:48 -> 15:03 插入97000+个节点，920000条边
class DRKGData:
    def __init__(self) -> None:
        path='./drkg.tsv'
        columns=['head','relation','tail']
        self.df=self.getData(path,columns)
        #self.df=self.df.iloc[:10000,:]

        self.Graph=Graph('http://localhost:7474',auth=('neo4j',"Ww1043790919!"), name = 'neo4j')
        #print(self.Graph.schema.node_labels)
        #self.Graph.run("create index for (n:Gene) on (n.id)")
        #print(self.Graph.schema.get_indexes("Gene"))
        #self.Graph.schema.create_index("Gene","id")  


        self.delete_old() 
        print("delete old data done")
        self.name2id=self.getName2id() 
        print("insert node done")
        #print(self.name2entity)
        self.insert()
        print("done")



    def getName2id(self):
        
        nodeid=0
        name2id=dict()
        
        heads=set(self.df.iloc[:,0])
        tails=set(self.df.iloc[:,2])
        nodes=heads|tails

        for node in nodes:
            entity=Node(node.split("::")[0],id=nodeid,name=node)
            self.Graph.create(entity)
            name2id[node]=nodeid
            nodeid+=1


        return name2id
    def delete_old(self):
       self.Graph.delete_all()

    def insert(self):
        

        matcher=NodeMatcher(self.Graph)
        
        for index,row in self.df.iterrows():
            headid=self.name2id[row['head']]
            tailid=self.name2id[row['tail']]
            
            head=matcher.match(id=headid).first()
            tail=matcher.match(id=tailid).first()
            rel=Relationship(head,row["relation"],tail)
            
            self.Graph.create(rel)


    def getData(self,path,columns):
        df=pd.read_csv(path,sep='\t')
        df.columns=columns
        return df



data=DRKGData()
