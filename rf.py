import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing on Knowledge Graph Embedding',
    )

    parser.add_argument('--kgname', type=str,help='kg name')
    parser.add_argument('--kg_kge_name', type=str, help='kg_kge name')
    

    return parser.parse_args(args)



def preprocess_data(entity_emb, entity_mapping, data):

    data['drug_feat'] = data['drug'].map(lambda x : entity_emb[entity_mapping[x]] if x in entity_mapping else None)
    data['target_feat'] = data['target'].map(lambda x : entity_emb[entity_mapping[x]] if x in entity_mapping else None)

    data = data[~data['drug_feat'].isna()]
    data = data[~data['target_feat'].isna()]
    data['feat'] = data.apply(lambda row: row['drug_feat'] + row['target_feat'], axis=1)

    X = np.array(data['feat'].tolist())

    # 提取标签
    y = data['relation'].values

    return X, y




def main(args):
    
    results_df = pd.DataFrame(columns=['Iteration', 'Accuracy', 'F1 Score', 'ROC AUC', 'Average Precision'])

    for i in range(5):
        # 预处理数据
        train_df = pd.read_csv(f'./rfdata/drugbank/train_df_{i}.csv')
        test_df = pd.read_csv(f'./rfdata/drugbank/test_df_{i}.csv')
        valid_df = pd.read_csv(f'./rfdata/drugbank/valid_df_{i}.csv')
        # train_df.columns = test_df.columns = valid_df.columns = ['drug','relation','ind']

        try:    
            entity_emb = np.load(f'./cskg/kgmodel/{args.kg_kge_name}/repo_{i}/entity_embedding_{i}.npy')
            # entity_emb.shape
            entitty_dict = pd.read_csv(f'./kgdata/{args.kgname}/entities_fix_{i}.dict',sep='\t',header=None)

            entity_mapping = dict(zip(entitty_dict[1], entitty_dict[0]))
            X_train, y_train = preprocess_data(entity_emb, entity_mapping,train_df)
            X_test, y_test = preprocess_data(entity_emb, entity_mapping,test_df)
            # X_valid, y_valid = preprocess_data(entity_emb, entity_mapping,valid_df)


            # 使用Logistic Regression分类器
            clf = RandomForestClassifier()

            clf.fit(X_train, y_train)

            # 在测试集上预测
            y_pred_test = clf.predict(X_test)
            y_pred_proba_test = clf.predict_proba(X_test)[:, 1]

            # 评估测试集性能
            accuracy_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test)
            roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
            average_precision_test = average_precision_score(y_test, y_pred_proba_test)

            print("Test set accuracy: {:.2f}%".format(accuracy_test * 100))
            print("Test set F1 score: {:.2f}".format(f1_test))
            print("Test set ROC AUC score: {:.2f}".format(roc_auc_test))
            print("Test set Average Precision score: {:.2f}".format(average_precision_test))

            # 计算ROC曲线和PR曲线的数据
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
            # 保存结果到DataFrame
            # results_df = results_df.append({'Iteration': i, 'Accuracy': accuracy_test,
            #                                 'F1 Score': f1_test, 'ROC AUC': roc_auc_test,
            #                                 'Average Precision': average_precision_test}, ignore_index=True)
            results_df = pd.concat([results_df, pd.DataFrame({
                    'Iteration': [i],
                    'Accuracy': [accuracy_test],
                    'F1 Score': [f1_test],
                    'ROC AUC': [roc_auc_test],
                    'Average Precision': [average_precision_test]
                })], ignore_index=True)

            # 绘制ROC曲线
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_test)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            # plt.savefig(f'./result/phkg_AutoSF_roc_curve_{i}.png')
            plt.savefig(f'./result/{args.kg_kge_name}_roc_curve_{i}.png')
            # plt.show()

            # 绘制PR曲线
            plt.figure()
            plt.plot(recall, precision, label='PR curve (area = %0.2f)' % average_precision_test)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")
            # 保存PR曲线为图片文件
            plt.savefig(f'./result/{args.kg_kge_name}_pr_curve_{i}.png')
            # plt.show()

            df = pd.DataFrame({'pre': y_test, 'label': y_pred_proba_test})
            df.to_csv(f'./result/{args.kg_kge_name}_result_{i}.csv')
        except:
            next




    # 将结果保存到CSV文件
    results_df.to_csv(f'drugbank_rf_{args.kg_kge_name}_results.csv', index=False)




if __name__ == '__main__':
    # 创建一个空DataFrame来存储结果

    main(parse_args())


