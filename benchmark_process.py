import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 读取交互三元组数据
inter_triple = pd.read_csv('./cid/drugbank.csv')
inter_triple['label'] = 1
inter_triple.columns = ['drug', 'protein','label' ]

# 获取所有节点（药物和疾病）
drugs = inter_triple['drug'].unique()
protein = inter_triple['protein'].unique()

# 设置交叉验证的折数
num_folds = 5

# 使用KFold进行交叉验证划分
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 生成所有的训练集和验证集索引
all_train_indices = []
all_valid_indices = []
for train_index, valid_index in kf.split(inter_triple):
    all_train_indices.append(train_index)
    all_valid_indices.append(valid_index)

for i in range(num_folds):
    train_index = all_train_indices[i]
    valid_index = all_valid_indices[i]

    train_data = inter_triple.iloc[train_index]
    valid_data = inter_triple.iloc[valid_index]

    # 测试集划分
    kf_valid = KFold(n_splits=2, shuffle=True, random_state=42)
    for valid_index, test_index in kf_valid.split(valid_data):
        valid_data = valid_data.iloc[valid_index]
        test_data = inter_triple.iloc[test_index]  # 使用 inter_triple 来获取测试数据
        break

    # 确保测试集和验证集中没有训练集中未出现的节点
    train_drugs = train_data['drug'].unique()
    train_protein = train_data['protein'].unique()

    test_data = test_data[(test_data['drug'].isin(train_drugs)) & (test_data['protein'].isin(train_protein))]
    valid_data = valid_data[(valid_data['drug'].isin(train_drugs)) & (valid_data['protein'].isin(train_protein))]

    # 保存划分后的数据
    np.savetxt(f"./benchmark/drugbank/train_data_{i}.txt", train_data.values, fmt="%s", delimiter=",")
    np.savetxt(f"./benchmark/drugbank/test_data_{i}.txt", test_data.values, fmt="%s", delimiter=",")
    np.savetxt(f"./benchmark/drugbank/valid_data_{i}.txt", valid_data.values, fmt="%s", delimiter=",")




#############################################################################################


def gene_hetionet_kg(i):
    kg_train = pd.read_csv('./kgsplit/hetionet_train_df.csv')
    kg_test = pd.read_csv('./kgsplit/hetionet_test_df.csv')
    kg_valid = pd.read_csv('./kgsplit/hetionet_valid_df.csv')


    inter_train = pd.read_csv(f'./benchmark/drugbank/train_data_{i}.txt',header=None)
    inter_test = pd.read_csv(f'./benchmark/drugbank/test_data_{i}.txt',header=None)
    inter_valid = pd.read_csv(f'./benchmark/drugbank/valid_data_{i}.txt',header=None)
    
    inter_train = inter_train[[0,2,1]]
    inter_test = inter_test[[0,2,1]]
    inter_valid = inter_valid[[0,2,1]]


    inter_train[2] = 'Compound-Gene'
    inter_train.columns = kg_train.columns
    

    inter_test[2] = 'Compound-Gene'
    inter_test.columns = kg_train.columns
    
    inter_valid[2] = 'Compound-Gene'
    inter_valid.columns = kg_train.columns
    
    kg_train = pd.concat([kg_train,inter_train])    
    kg_test = pd.concat([kg_test,inter_test])
    kg_valid = pd.concat([kg_valid,inter_valid])

    # 在kg_train和kg_test之间执行merge操作，并标记差异
    merged_train_test = kg_train.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_train中删除与kg_test完全相同的数据
    kg_train = merged_train_test[merged_train_test['_merge'] != 'both'][kg_train.columns]

    # 在kg_valid和kg_test之间执行merge操作，并标记差异
    merged_valid_test = kg_valid.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_valid中删除与kg_test完全相同的数据
    kg_valid = merged_valid_test[merged_valid_test['_merge'] != 'both'][kg_valid.columns]

    
    kg_train.to_csv(f'./kgdata/hetionet/train_{i}.txt',header=None,index=None,sep='\t')
    kg_test.to_csv(f'./kgdata/hetionet/test_{i}.txt',header=None,index=None,sep='\t')
    kg_valid.to_csv(f'./kgdata/hetionet/valid_{i}.txt',header=None,index=None,sep='\t')
    
    
    entities = set(
    list(kg_train['head']) + list(kg_train['tail']) 
    + list(kg_test['head']) + list(kg_test['tail']) 
    + list(kg_valid['head']) + list(kg_valid['tail']))
    relation = set(list(kg_train['relation'])+list(kg_test['relation'])+list(kg_valid['relation']))
    
    
    relation_df = pd.DataFrame(relation).reset_index()
    relation_df.to_csv(f"./kgdata/hetionet/relations_fix1.dict",header=None,index=None,sep='\t')
    
    entities_df = pd.DataFrame(entities).reset_index()
    entities_df.to_csv(f"./kgdata/hetionet/entities_fix_{i}.dict",header=None,index=None,sep='\t')

for i in range(5):
    gene_hetionet_kg(i)

#############################################################################################
    


def gene_biokg_kg(i):
    kg_train = pd.read_csv('./kgsplit/biokg_train_df.csv')
    kg_test = pd.read_csv('./kgsplit/biokg_test_df.csv')
    kg_valid = pd.read_csv('./kgsplit/biokg_valid_df.csv')
    # kg_train['relation'].value_counts()


    inter_train = pd.read_csv(f'./benchmark/drugbank/train_data_{i}.txt',header=None)
    inter_test = pd.read_csv(f'./benchmark/drugbank/test_data_{i}.txt',header=None)
    inter_valid = pd.read_csv(f'./benchmark/drugbank/valid_data_{i}.txt',header=None)
    inter_train = inter_train[[0,2,1]]
    inter_test = inter_test[[0,2,1]]
    inter_valid = inter_valid[[0,2,1]]


    inter_train[2] = 'drug_protein'
    inter_train.columns = kg_train.columns
    

    inter_test[2] = 'drug_protein'
    inter_test.columns = kg_train.columns
    
    inter_valid[2] = 'drug_protein'
    inter_valid.columns = kg_train.columns
    
    kg_train = pd.concat([kg_train,inter_train])    
    kg_test = pd.concat([kg_test,inter_test])
    kg_valid = pd.concat([kg_valid,inter_valid])

    # 在kg_train和kg_test之间执行merge操作，并标记差异
    merged_train_test = kg_train.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_train中删除与kg_test完全相同的数据
    kg_train = merged_train_test[merged_train_test['_merge'] != 'both'][kg_train.columns]

    # 在kg_valid和kg_test之间执行merge操作，并标记差异
    merged_valid_test = kg_valid.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_valid中删除与kg_test完全相同的数据
    kg_valid = merged_valid_test[merged_valid_test['_merge'] != 'both'][kg_valid.columns]

    
    kg_train.to_csv(f'./kgdata/biokg/train_{i}.txt',header=None,index=None,sep='\t')
    kg_test.to_csv(f'./kgdata/biokg/test_{i}.txt',header=None,index=None,sep='\t')
    kg_valid.to_csv(f'./kgdata/biokg/valid_{i}.txt',header=None,index=None,sep='\t')

    entities = set(
    list(kg_train['head']) + list(kg_train['tail']) 
    + list(kg_test['head']) + list(kg_test['tail']) 
    + list(kg_valid['head']) + list(kg_valid['tail']))
    relation = set(list(kg_train['relation'])+list(kg_test['relation'])+list(kg_valid['relation']))
    
    
    relation_df = pd.DataFrame(relation).reset_index()
    relation_df.to_csv(f"./kgdata/biokg/relations_fix1.dict",header=None,index=None,sep='\t')
    
    entities_df = pd.DataFrame(entities).reset_index()
    entities_df.to_csv(f"./kgdata/biokg/entities_fix_{i}.dict",header=None,index=None,sep='\t')

for i in range(5):
    gene_biokg_kg(i)



#############################################################################################
    

def gene_iBKH_kg(i):
    kg_train = pd.read_csv('./kgsplit/iBKH_train_df.csv')
    kg_test = pd.read_csv('./kgsplit/iBKH_test_df.csv')
    kg_valid = pd.read_csv('./kgsplit/iBKH_valid_df.csv')
    # kg_train['relation'].value_counts()

    inter_train = pd.read_csv(f'./benchmark/drugbank/train_data_{i}.txt',header=None)
    inter_test = pd.read_csv(f'./benchmark/drugbank/test_data_{i}.txt',header=None)
    inter_valid = pd.read_csv(f'./benchmark/drugbank/valid_data_{i}.txt',header=None)
    inter_train = inter_train[[0,2,1]]
    inter_test = inter_test[[0,2,1]]
    inter_valid = inter_valid[[0,2,1]]


    inter_train[2] = 'Drug-Gene'
    inter_train.columns = kg_train.columns
    

    inter_test[2] = 'Drug-Genee'
    inter_test.columns = kg_train.columns
    
    inter_valid[2] = 'Drug-Gene'
    inter_valid.columns = kg_train.columns
    
    kg_train = pd.concat([kg_train,inter_train])    
    kg_test = pd.concat([kg_test,inter_test])
    kg_valid = pd.concat([kg_valid,inter_valid])

    # 在kg_train和kg_test之间执行merge操作，并标记差异
    merged_train_test = kg_train.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_train中删除与kg_test完全相同的数据
    kg_train = merged_train_test[merged_train_test['_merge'] != 'both'][kg_train.columns]

    # 在kg_valid和kg_test之间执行merge操作，并标记差异
    merged_valid_test = kg_valid.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_valid中删除与kg_test完全相同的数据
    kg_valid = merged_valid_test[merged_valid_test['_merge'] != 'both'][kg_valid.columns]

    
    kg_train.to_csv(f'./kgdata/iBKH/train_{i}.txt',header=None,index=None,sep='\t')
    kg_test.to_csv(f'./kgdata/iBKH/test_{i}.txt',header=None,index=None,sep='\t')
    kg_valid.to_csv(f'./kgdata/iBKH/valid_{i}.txt',header=None,index=None,sep='\t')

    entities = set(
    list(kg_train['head']) + list(kg_train['tail']) 
    + list(kg_test['head']) + list(kg_test['tail']) 
    + list(kg_valid['head']) + list(kg_valid['tail']))
    relation = set(list(kg_train['relation'])+list(kg_test['relation'])+list(kg_valid['relation']))
    
    
    relation_df = pd.DataFrame(relation).reset_index()
    relation_df.to_csv(f"./kgdata/iBKH/relations_fix1.dict",header=None,index=None,sep='\t')
    
    entities_df = pd.DataFrame(entities).reset_index()
    entities_df.to_csv(f"./kgdata/iBKH/entities_fix_{i}.dict",header=None,index=None,sep='\t')

for i in range(5):
    gene_iBKH_kg(i)

#############################################################################################
    


def gene_phkg_kg(i):
    kg_train = pd.read_csv('./kgsplit/phkg_train_df.csv')
    kg_test = pd.read_csv('./kgsplit/phkg_test_df.csv')
    kg_valid = pd.read_csv('./kgsplit/phkg_valid_df.csv')
    # kg_train['relation'].value_counts()


    inter_train = pd.read_csv(f'./benchmark/drugbank/train_data_{i}.txt',header=None)
    inter_test = pd.read_csv(f'./benchmark/drugbank/test_data_{i}.txt',header=None)
    inter_valid = pd.read_csv(f'./benchmark/drugbank/valid_data_{i}.txt',header=None)
    inter_train = inter_train[[0,2,1]]
    inter_test = inter_test[[0,2,1]]
    inter_valid = inter_valid[[0,2,1]]


    inter_train[2] = 'chemical-gene'
    inter_train.columns = kg_train.columns
    

    inter_test[2] = 'chemical-gene'
    inter_test.columns = kg_train.columns
    
    inter_valid[2] = 'chemical-gene'
    inter_valid.columns = kg_train.columns
    
    kg_train = pd.concat([kg_train,inter_train])    
    kg_test = pd.concat([kg_test,inter_test])
    kg_valid = pd.concat([kg_valid,inter_valid])

    # 在kg_train和kg_test之间执行merge操作，并标记差异
    merged_train_test = kg_train.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_train中删除与kg_test完全相同的数据
    kg_train = merged_train_test[merged_train_test['_merge'] != 'both'][kg_train.columns]

    # 在kg_valid和kg_test之间执行merge操作，并标记差异
    merged_valid_test = kg_valid.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_valid中删除与kg_test完全相同的数据
    kg_valid = merged_valid_test[merged_valid_test['_merge'] != 'both'][kg_valid.columns]

    
    kg_train.to_csv(f'./kgdata/phkg/train_{i}.txt',header=None,index=None,sep='\t')
    kg_test.to_csv(f'./kgdata/phkg/test_{i}.txt',header=None,index=None,sep='\t')
    kg_valid.to_csv(f'./kgdata/phkg/valid_{i}.txt',header=None,index=None,sep='\t')

    entities = set(
    list(kg_train['head']) + list(kg_train['tail']) 
    + list(kg_test['head']) + list(kg_test['tail']) 
    + list(kg_valid['head']) + list(kg_valid['tail']))
    relation = set(list(kg_train['relation'])+list(kg_test['relation'])+list(kg_valid['relation']))
    
    
    relation_df = pd.DataFrame(relation).reset_index()
    relation_df.to_csv(f"./kgdata/phkg/relations_fix1.dict",header=None,index=None,sep='\t')
    
    entities_df = pd.DataFrame(entities).reset_index()
    entities_df.to_csv(f"./kgdata/phkg/entities_fix_{i}.dict",header=None,index=None,sep='\t')

for i in range(5):
    gene_phkg_kg(i)

#############################################################################################
    


def gene_primekg_kg(i):

    kg_train = pd.read_csv('./kgsplit/primekg_train_df.csv')
    kg_test = pd.read_csv('./kgsplit/primekg_test_df.csv')
    kg_valid = pd.read_csv('./kgsplit/primekg_valid_df.csv')
    # kg_train['relation'].value_counts()

    inter_train = pd.read_csv(f'./benchmark/drugbank/train_data_{i}.txt',header=None)
    inter_test = pd.read_csv(f'./benchmark/drugbank/test_data_{i}.txt',header=None)
    inter_valid = pd.read_csv(f'./benchmark/drugbank/valid_data_{i}.txt',header=None)
    inter_train = inter_train[[0,2,1]]
    inter_test = inter_test[[0,2,1]]
    inter_valid = inter_valid[[0,2,1]]


    inter_train[2] = 'drug_protein'
    inter_train.columns = ['head', 'relation', 'tail']
    

    inter_test[2] = 'drug_protein'
    inter_test.columns = ['head', 'relation', 'tail']
    
    inter_valid[2] = 'drug_protein'
    inter_valid.columns = ['head', 'relation', 'tail']
    
    kg_train = pd.concat([kg_train,inter_train])    
    kg_test = pd.concat([kg_test,inter_test])
    kg_valid = pd.concat([kg_valid,inter_valid])

    # 在kg_train和kg_test之间执行merge操作，并标记差异
    merged_train_test = kg_train.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_train中删除与kg_test完全相同的数据
    kg_train = merged_train_test[merged_train_test['_merge'] != 'both'][kg_train.columns]

    # 在kg_valid和kg_test之间执行merge操作，并标记差异
    merged_valid_test = kg_valid.merge(kg_test, on=['head', 'relation', 'tail'], how='left', indicator=True)

    # 从kg_valid中删除与kg_test完全相同的数据
    kg_valid = merged_valid_test[merged_valid_test['_merge'] != 'both'][kg_valid.columns]

    
    kg_train.to_csv(f'./kgdata/primekg/train_{i}.txt',header=None,index=None,sep='\t')
    kg_test.to_csv(f'./kgdata/primekg/test_{i}.txt',header=None,index=None,sep='\t')
    kg_valid.to_csv(f'./kgdata/primekg/valid_{i}.txt',header=None,index=None,sep='\t')

    entities = set(
    list(kg_train['head']) + list(kg_train['tail']) 
    + list(kg_test['head']) + list(kg_test['tail']) 
    + list(kg_valid['head']) + list(kg_valid['tail']))
    relation = set(list(kg_train['relation'])+list(kg_test['relation'])+list(kg_valid['relation']))
    
    
    relation_df = pd.DataFrame(relation).reset_index()
    relation_df.to_csv(f"./kgdata/primekg/relations_fix1.dict",header=None,index=None,sep='\t')
    
    entities_df = pd.DataFrame(entities).reset_index()
    entities_df.to_csv(f"./kgdata/primekg/entities_fix_{i}.dict",header=None,index=None,sep='\t')

for i in range(5):
    gene_primekg_kg(i)
