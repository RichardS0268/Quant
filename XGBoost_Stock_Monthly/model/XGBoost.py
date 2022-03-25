import time
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance,plot_tree, train
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold #交叉验证
import os
import seaborn as sns

train_file = ["2010.csv", "2011.csv", "2012.csv", "2013.csv", "2014.csv", "2015.csv", "2016.csv", "2017.csv", "2018.csv", "2019.csv", "2020.csv", "2021.csv"]

features = ['Capitalization', 'CirculatingCap','CirculatingMarketCap', 'MarketCap', 'PBRatio', 'PCFRatio', 'PSRatio','PeRatio', 'PeRatioLYR', 'TurnoverRatio']

for i in range(7):
    data = pd.read_csv(f"../stock_data/train_data/{train_file[i]}", index_col="index")
    train_data_sorted = data.sort_values(by = "rf", ascending=False)

    long_set_train = train_data_sorted.iloc[:int(train_data_sorted.shape[0]*0.3), :]
    long_set_train["rf"] = 2
    plain_set_train = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.3):int(train_data_sorted.shape[0]*0.7), :]
    plain_set_train["rf"] = 1
    short_set_train = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.7):, :]
    short_set_train["rf"] = 0
    train_data = long_set_train.append(plain_set_train).append(short_set_train)

    for file in train_file[i+1:i+4]:
        data = pd.read_csv(f"../stock_data/train_data/{file}", index_col="index")
        train_data_sorted = data.sort_values(by = "rf", ascending=False)
        long_set = train_data_sorted.iloc[:int(train_data_sorted.shape[0]*0.3), :]
        long_set["rf"] = 2
        plain_set = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.3):int(train_data_sorted.shape[0]*0.7), :]
        plain_set["rf"] = 1 
        short_set = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.7):, :]
        short_set["rf"] = 0
        train_set =long_set.append(plain_set).append(short_set)
        train_data = train_data.append(train_set)

    #为保证回测阶段的连续性——即每年的1月可以进行调仓，则在训练集中不能存在信息泄露。因此每段训练集中最后一年的最后一个月要去除，因为这个月的label是次年1月的超额回报率。
    data = pd.read_csv(f"../stock_data/train_data/{train_file[i+4]}", index_col="index")
    data = data.drop(index=data[data["time"]==train_file[i+4][:4]+"-12"].index)
    train_data_sorted = data.sort_values(by = "rf", ascending=False)
    long_set = train_data_sorted.iloc[:int(train_data_sorted.shape[0]*0.3), :]
    long_set["rf"] = 2
    plain_set = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.3):int(train_data_sorted.shape[0]*0.7), :]
    plain_set["rf"] = 1 
    short_set = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.7):, :]
    short_set["rf"] = 0
    train_set =long_set.append(plain_set).append(short_set)
    train_data = train_data.append(train_set)


    test_data = pd.read_csv(f"../stock_data/train_data/{train_file[i+5]}", index_col="index")
    test_data_sorted = data.sort_values(by = "rf", ascending=False)
    long_set_test = test_data_sorted.iloc[:int(test_data_sorted.shape[0]*0.3), :]
    long_set_test["rf"] = 2
    plain_set_test = test_data_sorted.iloc[int(test_data_sorted.shape[0]*0.3):int(test_data_sorted.shape[0]*0.7), :]
    plain_set_test["rf"] = 1
    short_set_test = test_data_sorted.iloc[int(test_data_sorted.shape[0]*0.7):, :]
    short_set_test["rf"] = 0
    test_data =long_set_test.append(plain_set_test).append(short_set_test)

    # 加载样本数据集，划分训练集，测试集
    X_train_set = train_data[features]
    y_train_set = train_data["rf"]
    X_train, X_val, y_train, y_val = train_test_split(X_train_set, y_train_set, test_size=0.1, random_state=1234565)
    X_test = test_data[features]
    y_test = test_data["rf"]

    # 数据集格式转换
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, feature_names=features)

    # 训练算法参数设置
    params = {
        # 通用参数
        'booster': 'gbtree', # 使用的弱学习器,有两种选择gbtree（默认）和gblinear,gbtree是基于
                            # 树模型的提升计算，gblinear是基于线性模型的提升计算
        # 'nthread' XGBoost运行时的线程数，缺省时是当前系统获得的最大线程数
        'silent':0, # 0：表示打印运行时信息，1：表示以缄默方式运行，默认为0
        'num_feature':10, # boosting过程中使用的特征维数
        'seed': 1000, # 随机数种子
        # 任务参数
        'objective': 'multi:softmax', # 多分类的softmax,objective用来定义学习任务及相应的损失函数
        'num_class': 3, # 类别总数
        # 提升参数
        'gamma': 0.1, # 叶子节点进行划分时需要损失函数减少的最小值
        'max_depth': 6, # 树的最大深度，由交叉验证知，选6较为合适
        'lambda': 2, # 正则化权重
        'subsample': 1, # 训练模型的样本占总样本的比例，用于防止过拟合
        'colsample_bytree': 0.7, # 建立树时对特征进行采样的比例
        'min_child_weight': 3, # 叶子节点继续划分的最小的样本权重和
        'eta': 0.1, # 加法模型中使用的收缩步长   
        
    }
    num_rounds = 50
    plst = list(params.items())
    xgb_clf = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(X_test, feature_names=features)

    y_pred = xgb_clf.predict(dtest)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"model:{train_file[i]}-{train_file[i+4]} => accuarcy: %.2f%%" % (accuracy*100.0))

    # # 显示重要特征
    # plot_importance(xgb_clf)
    # plt.show()

    # 保存模型
    xgb_clf.save_model(f"model_set/xgb-{train_file[i][:4]}-{train_file[i+4][:4]}.json")






