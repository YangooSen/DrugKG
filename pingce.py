import pandas as pd
import random
from collections import defaultdict


##########################################################################################
##########################################################################################

hetionet = pd.read_csv('./kg/hetionet_kg_mapped.csv')
# hetionet['metaedge'].value_counts()
hetionet[hetionet['metaedge']=='Compound-Gene']
hetionet.columns=['head', 'relation', 'tail']


# 将数据框转换为列表形式，以便后续处理
triples = [(row['head'], row['relation'], row['tail']) for _, row in hetionet.iterrows()]
# 统计关系出现次数
relation_count = defaultdict(int)
for h, r, t in triples:
    relation_count[r] += 1

# 确定数据集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
num_triples = len(triples)
num_train = int(num_triples * train_ratio)
num_valid = int(num_triples * valid_ratio)
num_test = num_triples - num_train - num_valid

# 划分数据集
train_triples = []
valid_triples = []
test_triples = []


## 遍历所有关系，对每个关系划分train test valid ，将不同关系放到一个列表中
for r in relation_count.keys():
    # 获取关系r对应的三元组列表
    r_triples = [(h, r, t) for h, r_, t in triples if r == r_]
    
    # 随机打乱关系r的三元组
    random.shuffle(r_triples)
    
    # 计算关系r在train、test和valid数据集中分配的三元组数量
    num_r_train = int(len(r_triples) * train_ratio)
    num_r_valid = int(len(r_triples) * valid_ratio)
    num_r_test = len(r_triples) - num_r_train - num_r_valid
    
    # 分配三元组到不同数据集中
    train_triples.extend(r_triples[:num_r_train])
    valid_triples.extend(r_triples[num_r_train:num_r_train+num_r_valid])
    test_triples.extend(r_triples[num_r_train+num_r_valid:num_r_train+num_r_valid+num_r_test])

# 确保每个数据集中的三元组数量均匀分布
random.shuffle(train_triples)
random.shuffle(valid_triples)
random.shuffle(test_triples)

# 将划分后的三元组转换回数据框
train_df = pd.DataFrame(train_triples, columns=['head', 'relation', 'tail'])
valid_df = pd.DataFrame(valid_triples, columns=['head', 'relation', 'tail'])
test_df = pd.DataFrame(test_triples, columns=['head', 'relation', 'tail'])

train_df.to_csv('./kgsplit/hetionet_train_df.csv',index=None)
valid_df.to_csv('./kgsplit/hetionet_valid_df.csv',index=None)
test_df.to_csv('./kgsplit/hetionet_test_df.csv',index=None)


##########################################################################################
##########################################################################################

biokg = pd.read_csv('./kg/biokg_mapped.csv')
biokg['relation'].value_counts()
biokg.columns = ['head', 'relation', 'tail']

# biokg[biokg['relation'].isna()]


# 将数据框转换为列表形式，以便后续处理
triples = [(row['head'], row['relation'], row['tail']) for _, row in biokg.iterrows()]
# 统计关系出现次数
relation_count = defaultdict(int)
for h, r, t in triples:
    relation_count[r] += 1

# 确定数据集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
num_triples = len(triples)
num_train = int(num_triples * train_ratio)
num_valid = int(num_triples * valid_ratio)
num_test = num_triples - num_train - num_valid

# 划分数据集
train_triples = []
valid_triples = []
test_triples = []



## 遍历所有关系，对每个关系划分train test valid ，将不同关系放到一个列表中
for r in relation_count.keys():
    # 获取关系r对应的三元组列表
    r_triples = [(h, r, t) for h, r_, t in triples if r == r_]
    
    # 随机打乱关系r的三元组
    random.shuffle(r_triples)
    
    # 计算关系r在train、test和valid数据集中分配的三元组数量
    num_r_train = int(len(r_triples) * train_ratio)
    num_r_valid = int(len(r_triples) * valid_ratio)
    num_r_test = len(r_triples) - num_r_train - num_r_valid
    
    # 分配三元组到不同数据集中
    train_triples.extend(r_triples[:num_r_train])
    valid_triples.extend(r_triples[num_r_train:num_r_train+num_r_valid])
    test_triples.extend(r_triples[num_r_train+num_r_valid:num_r_train+num_r_valid+num_r_test])

# 确保每个数据集中的三元组数量均匀分布
random.shuffle(train_triples)
random.shuffle(valid_triples)
random.shuffle(test_triples)

# 将划分后的三元组转换回数据框
train_df = pd.DataFrame(train_triples, columns=['head', 'relation', 'tail'])
valid_df = pd.DataFrame(valid_triples, columns=['head', 'relation', 'tail'])
test_df = pd.DataFrame(test_triples, columns=['head', 'relation', 'tail'])

train_df.to_csv('./kgsplit/biokg_train_df.csv',index=None)
valid_df.to_csv('./kgsplit/biokg_valid_df.csv',index=None)
test_df.to_csv('./kgsplit/biokg_test_df.csv',index=None)



##########################################################################################

iBKH = pd.read_csv('./kg/iBKH_mapped.csv')
iBKH['relation'].value_counts()


# 将数据框转换为列表形式，以便后续处理
triples = [(row['head'], row['relation'], row['tail']) for _, row in iBKH.iterrows()]
# 统计关系出现次数
relation_count = defaultdict(int)
for h, r, t in triples:
    relation_count[r] += 1

# 确定数据集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
num_triples = len(triples)
num_train = int(num_triples * train_ratio)
num_valid = int(num_triples * valid_ratio)
num_test = num_triples - num_train - num_valid

# 划分数据集
train_triples = []
valid_triples = []
test_triples = []



## 遍历所有关系，对每个关系划分train test valid ，将不同关系放到一个列表中
for r in relation_count.keys():
    # 获取关系r对应的三元组列表
    r_triples = [(h, r, t) for h, r_, t in triples if r == r_]
    
    # 随机打乱关系r的三元组
    random.shuffle(r_triples)
    
    # 计算关系r在train、test和valid数据集中分配的三元组数量
    num_r_train = int(len(r_triples) * train_ratio)
    num_r_valid = int(len(r_triples) * valid_ratio)
    num_r_test = len(r_triples) - num_r_train - num_r_valid
    
    # 分配三元组到不同数据集中
    train_triples.extend(r_triples[:num_r_train])
    valid_triples.extend(r_triples[num_r_train:num_r_train+num_r_valid])
    test_triples.extend(r_triples[num_r_train+num_r_valid:num_r_train+num_r_valid+num_r_test])

# 确保每个数据集中的三元组数量均匀分布
random.shuffle(train_triples)
random.shuffle(valid_triples)
random.shuffle(test_triples)

# 将划分后的三元组转换回数据框
train_df = pd.DataFrame(train_triples, columns=['head', 'relation', 'tail'])
valid_df = pd.DataFrame(valid_triples, columns=['head', 'relation', 'tail'])
test_df = pd.DataFrame(test_triples, columns=['head', 'relation', 'tail'])

train_df.to_csv('./kgsplit/iBKH_train_df.csv',index=None)
valid_df.to_csv('./kgsplit/iBKH_valid_df.csv',index=None)
test_df.to_csv('./kgsplit/iBKH_test_df.csv',index=None)



##########################################################################################

phkg = pd.read_csv('./kg/phkg_mapped.csv')
phkg = phkg[['head', 'relation', 'tail']]
phkg['relation'].value_counts()
phkg[phkg['relation'].isna()]

# 将 'gene-chemical' 的 'head' 和 'tail' 互换, 将 'gene-chemical' 改成 'chemical-gene'
phkg.loc[phkg['relation'] == 'gene-chemical', ['head', 'tail']] = phkg.loc[phkg['relation'] == 'gene-chemical', ['tail', 'head']].values
phkg.loc[phkg['relation'] == 'gene-chemical', 'relation'] = 'chemical-gene'

# 将 'disease-chemical' 的 'head' 和 'tail' 互换, 将 'disease-chemical' 改成 'chemical-disease'
phkg.loc[phkg['relation'] == 'disease-chemical', ['head', 'tail']] = phkg.loc[phkg['relation'] == 'disease-chemical', ['tail', 'head']].values
phkg[phkg['relation'] == 'disease-chemical']
phkg.loc[phkg['relation'] == 'disease-chemical', 'relation'] = 'chemical-disease'

# 将 'disease-gene' 的 'head' 和 'tail' 互换, 将 'disease-gene' 改成 'gene-disease'
phkg.loc[phkg['relation'] == 'disease-gene', ['head', 'tail']] = phkg.loc[phkg['relation'] == 'disease-gene', ['tail', 'head']].values
phkg[phkg['relation'] == 'disease-gene']
phkg.loc[phkg['relation'] == 'disease-gene', 'relation'] = 'gene-disease'

phkg.to_csv('./kg/phkg_mapped01.csv',index=None)



phkg['relation'].value_counts()


# 将数据框转换为列表形式，以便后续处理
triples = [(row['head'], row['relation'], row['tail']) for _, row in phkg.iterrows()]
# 统计关系出现次数
relation_count = defaultdict(int)
for h, r, t in triples:
    relation_count[r] += 1

# 确定数据集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
num_triples = len(triples)
num_train = int(num_triples * train_ratio)
num_valid = int(num_triples * valid_ratio)
num_test = num_triples - num_train - num_valid

# 划分数据集
train_triples = []
valid_triples = []
test_triples = []



## 遍历所有关系，对每个关系划分train test valid ，将不同关系放到一个列表中
for r in relation_count.keys():
    # 获取关系r对应的三元组列表
    r_triples = [(h, r, t) for h, r_, t in triples if r == r_]
    
    # 随机打乱关系r的三元组
    random.shuffle(r_triples)
    
    # 计算关系r在train、test和valid数据集中分配的三元组数量
    num_r_train = int(len(r_triples) * train_ratio)
    num_r_valid = int(len(r_triples) * valid_ratio)
    num_r_test = len(r_triples) - num_r_train - num_r_valid
    
    # 分配三元组到不同数据集中
    train_triples.extend(r_triples[:num_r_train])
    valid_triples.extend(r_triples[num_r_train:num_r_train+num_r_valid])
    test_triples.extend(r_triples[num_r_train+num_r_valid:num_r_train+num_r_valid+num_r_test])

# 确保每个数据集中的三元组数量均匀分布
random.shuffle(train_triples)
random.shuffle(valid_triples)
random.shuffle(test_triples)

# 将划分后的三元组转换回数据框
train_df = pd.DataFrame(train_triples, columns=['head', 'relation', 'tail'])
valid_df = pd.DataFrame(valid_triples, columns=['head', 'relation', 'tail'])
test_df = pd.DataFrame(test_triples, columns=['head', 'relation', 'tail'])

train_df.to_csv('./kgsplit/phkg_train_df.csv',index=None)
valid_df.to_csv('./kgsplit/phkg_valid_df.csv',index=None)
test_df.to_csv('./kgsplit/phkg_test_df.csv',index=None)



##########################################################################################

primekg = pd.read_csv('./kg/primekg_mapped.csv')
primekg = primekg[['head', 'relation', 'tail']]
primekg['relation'].value_counts()
primekg[primekg['relation'].isna()]

# 将数据框转换为列表形式，以便后续处理
triples = [(row['head'], row['relation'], row['tail']) for _, row in primekg.iterrows()]
# 统计关系出现次数
relation_count = defaultdict(int)
for h, r, t in triples:
    relation_count[r] += 1

# 确定数据集比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 计算数据集大小
num_triples = len(triples)
num_train = int(num_triples * train_ratio)
num_valid = int(num_triples * valid_ratio)
num_test = num_triples - num_train - num_valid

# 划分数据集
train_triples = []
valid_triples = []
test_triples = []



## 遍历所有关系，对每个关系划分train test valid ，将不同关系放到一个列表中
for r in relation_count.keys():
    # 获取关系r对应的三元组列表
    r_triples = [(h, r, t) for h, r_, t in triples if r == r_]
    
    # 随机打乱关系r的三元组
    random.shuffle(r_triples)
    
    # 计算关系r在train、test和valid数据集中分配的三元组数量
    num_r_train = int(len(r_triples) * train_ratio)
    num_r_valid = int(len(r_triples) * valid_ratio)
    num_r_test = len(r_triples) - num_r_train - num_r_valid
    
    # 分配三元组到不同数据集中
    train_triples.extend(r_triples[:num_r_train])
    valid_triples.extend(r_triples[num_r_train:num_r_train+num_r_valid])
    test_triples.extend(r_triples[num_r_train+num_r_valid:num_r_train+num_r_valid+num_r_test])

# 确保每个数据集中的三元组数量均匀分布
random.shuffle(train_triples)
random.shuffle(valid_triples)
random.shuffle(test_triples)

# 将划分后的三元组转换回数据框
train_df = pd.DataFrame(train_triples, columns=['head', 'relation', 'tail'])
valid_df = pd.DataFrame(valid_triples, columns=['head', 'relation', 'tail'])
test_df = pd.DataFrame(test_triples, columns=['head', 'relation', 'tail'])

train_df.to_csv('./kgsplit/primekg_train_df.csv',index=None)
valid_df.to_csv('./kgsplit/primekg_valid_df.csv',index=None)
test_df.to_csv('./kgsplit/primekg_test_df.csv',index=None)
