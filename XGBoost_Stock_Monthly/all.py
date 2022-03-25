import pandas as pd
from functools import reduce
from typing import Tuple, Sequence
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

stock_id = {}
M_start_date = [] # 每月第一个交易日
M_end_date = [] # 每月最后一个交易日
start_map = {} # 月份和该月最后一个交易日的映射
end_map = {} # 月份和该月第一个交易日的映射
features = [
 'Capitalization',
 'CirculatingCap',
 'CirculatingMarketCap',
 'MarketCap',
 'PBRatio',
 'PCFRatio',
 'PSRatio',
 'PeRatio',
 'PeRatioLYR',
 'TurnoverRatio']
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

monthly_acc = {} # 模型的预测准确率, 在predict()中使用

train_file = ["2010.csv", "2011.csv", "2012.csv", "2013.csv", "2014.csv", "2015.csv", "2016.csv", "2017.csv", "2018.csv", "2019.csv", "2020.csv", "2021.csv"]
# -----------------------------------------------------------------------
# == 全局数据集 == 
hs300 = pd.read_csv("hs300.csv") # 获取沪深300历史数据，数据来自JoinQuant
price = pd.read_pickle("./stock_price_standard_")
valuation = pd.read_pickle("./stock_valuation_standard_")
# -----------------------------------------------------------------------
# == 回测参数 ==
Cash = 10000000 # allocation = 1000w
largeCap_portfolio = {} # 大市值公司的仓位
smallCap_portfolio = {} # 小市值公司的仓位
portfolio_capacity = 10 # portfolio中的个股个数
largeCap_Account = {} # 每一期期末——调仓前一天largeCap_portfolio中的个股价值（持有量*当日收盘价）
smallCap_Account = {} # 每一期期末——调仓前一天smallCap_portfolio中的个股价值（持有量*当日收盘价）
total_Account = {} # 每一期期末——调仓前一天的账户总价值（Cash + largeCap_Account + smallCap_Account）
Cap_leverage = 0.2 # 资金的0.2流向大市值公司，剩下的0.8流向小市值公司
vol_rate = 0.0001 # 考虑冲击成本，回测时每次调仓的成交量应小于当日该个股总成交量的0.0001倍
tax_rate = 0.0014 # 交易费用按交易金额的0.14%处理


def data_generate()->None:
    '''
    + 构建训练样本，将超额收益（相对于沪深300）作为label，将valuation文件中的entry作为feature
    + 以月为单位，即每支股票每个月的features和下一个月的超额收益（label）作为一条训练数据
    + 以年为集合进行存储，文件存储在目录"./gross_data"下
    '''
    dt = price.stack()
    dt = dt.reset_index()
    single_stock = price.groupby(price.index)
    code_set = set(list(dt["level_0"]))
    num_id = [x for x in range(1, 1411)]
    stock_id = dict(zip(num_id, code_set))
    closePrice = single_stock.get_group((stock_id[1], "Close"))
    #获得股票收盘价
    for i in range(2, 1411):
        closePrice = closePrice.append(single_stock.get_group(((stock_id[i], "Close"))))
        openPrice = single_stock.get_group((stock_id[1], "Open"))
    #获得股票开盘价
    for i in range(2, 1411):
        openPrice = closePrice.append(single_stock.get_group(((stock_id[i], "Open"))))

    hs300.index = pd.to_datetime(hs300["Unnamed: 0"])
    hs300.drop("Unnamed: 0", axis = 1, inplace = True)
    #获得月初，月末数据,去除2022年2月的数据
    hs300_M_end = hs300.loc[hs300.groupby(hs300.index.to_period('M')).apply(lambda x: x.index.max())]
    hs300_M_start = hs300.loc[hs300.groupby(hs300.index.to_period('M')).apply(lambda x: x.index.min())]
    hs300_M_end = hs300_M_end[:-1]
    hs300_M_start = hs300_M_start[:-1]
    # 计算沪深300每月收益率
    mon = list(set(hs300.index))
    mon.sort()
    trade_mon =  [x.strftime("%Y-%m") for x in mon]
    func = lambda x,y:x if y in x else x + [y]
    trade_mon = reduce(func, [[], ] + trade_mon)
    hs300_mon = pd.DataFrame(columns=["trade_mon", "rf"])
    hs300_mon["trade_mon"] = trade_mon[:-1]
    hs300_mon["rf"] = [(endPrice - initPrice)/initPrice for initPrice, endPrice in zip(hs300_M_start["close"], hs300_M_end["close"])]
    # 计算每支股票超额收益率，作为label
    M_start_date = list(hs300_M_start.index)
    M_end_date = list(hs300_M_end.index)
    M_return = pd.DataFrame(columns=["num_id"]+list(hs300_mon["trade_mon"]))
    M_return["num_id"] = num_id
    for i in range(0, 145):
        M_return[list(hs300_mon["trade_mon"])[i]] = list((closePrice.loc[:, M_end_date[i]] - closePrice.loc[:, M_start_date[i]])/closePrice.loc[:, M_start_date[i]])-hs300_mon.loc[i, "rf"]

    stock_val = valuation.groupby(valuation.index)
    M_feature = pd.DataFrame(columns=M_end_date)
    for num in stock_id:
        for feature in features:
            M_feature = M_feature.append(stock_val.get_group((stock_id[num], feature))[M_end_date])

    for year in years:
        train_ = pd.DataFrame(columns=["num_id", "time"] + features + ["rf"])
        # 日期对应：feature是月末，照应的label是下个月的超额收益
        date = list(hs300_mon["trade_mon"])
        tmp1 = []
        for i in range(13):
            tmp1 = tmp1 + num_id
            train_["time"][i*1410: (i+1)*1410] = date[i+132]
        train_["num_id"] = tmp1
        if math.isnan(train_["time"][1]):
            for i in range(13):
                train_["time"][i*1410: (i+1)*1410] = date[i+132]
        
        date_map = dict(zip(date, M_end_date))
        date_map2 = dict(zip(date[:-1], date[1:]))
        for feature in features:
            for i in range(16920):
                train_[feature][i] = list(M_feature[date_map[train_["time"][i]]][M_feature.index == (stock_id[train_["num_id"][i]], feature)])[0]

            for i in range(16920):
                train_["rf"][i] = M_return[date_map2[train_["time"][i]]][train_["num_id"][i]-1]
            train_ = train_.dropna(axis=0, how='any')
        train_.to_csv(f"./gross_data/{year}.csv")

def data_handler()->None:
    '''
    + 对gross data进行预处理，主要包含两部分
    + 1. 中位数去极值
    + 2. features数据归一化
    + 预处理后的数据保存在目录"./train_data"下
    '''
    for data_time in years:
        data = pd.read_csv(f"./gross_data/{data_time}.csv", index_col = 0)
        mons = list(set(list(data["time"])))
        features = list(set(data.columns[2:-1]))
        data.reset_index(inplace=True)
        DM = pd.DataFrame(columns=["time"]+features)
        DM["time"] = mons
        for i in range(len(mons)):
            for feature in features:
                DM[feature][i] = data[data["time"]==mons[i]][feature].median()
        DM1 = pd.DataFrame(columns=["time"]+features)
        DM1["time"] = mons
        for i in range(len(mons)):
            for feature in features:
                DM1[feature][i] = (data[data["time"]==mons[i]][feature]-float(DM[DM["time"]==mons[i]][feature])).abs().median()
        Range = pd.DataFrame(columns=["bond"]+features)
        Range["bond"] = ["lower_bond", "upper_bound"]
        for feature in features:
            for i in range(data.shape[0]):
                lb = float(DM[DM["time"] == data["time"][i]][feature]) - 5 * float(DM1[DM1["time"] == data["time"][i]][feature])
                ub = float(DM[DM["time"] == data["time"][i]][feature]) + 5 * float(DM1[DM1["time"] == data["time"][i]][feature])
                if data[feature][i] < lb:
                    data[feature][i] = lb
                if data[feature][i] > ub:
                    data[feature][i] = ub
        Mean = pd.DataFrame(columns=["time"]+features)
        Mean["time"] = mons
        st = pd.DataFrame(columns=["time"]+features)
        st["time"] = mons
        for i in range(len(mons)):
            for feature in features:
                Mean[feature][i] = data[data["time"]==mons[i]][feature].mean()
        for i in range(len(mons)):
            for feature in features:
                st[feature][i] = data[data["time"]==mons[i]][feature].std()
        for feature in features:
            for i in range(data.shape[0]):
                data[feature][i] = (data[feature][i] - Mean[Mean["time"]==data["time"][i]][feature])/st[st["time"]==data["time"][i]][feature]
        print(f"already {data_time}")
        data.to_csv(f"./train_data/{data_time}.csv")

def train_xgb_model()->None:
    '''
    + 批量训练XGBoost模型，超参数有网格搜索+交叉验证（GridCV）得出
    + 由于要进行7轮回测，因此共训练7个模型
    + 将训练好的模型参数保存在目录"./model_set"下
    '''
    for i in range(7):
        data = pd.read_csv(f"./train_data/{train_file[i]}", index_col="index")
        train_data_sorted = data.sort_values(by = "rf", ascending=False)

        long_set_train = train_data_sorted.iloc[:int(train_data_sorted.shape[0]*0.3), :]
        long_set_train["rf"] = 2
        plain_set_train = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.3):int(train_data_sorted.shape[0]*0.7), :]
        plain_set_train["rf"] = 1
        short_set_train = train_data_sorted.iloc[int(train_data_sorted.shape[0]*0.7):, :]
        short_set_train["rf"] = 0
        train_data = long_set_train.append(plain_set_train).append(short_set_train)

        for file in train_file[i+1:i+4]:
            data = pd.read_csv(f"./train_data/{file}", index_col="index")
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
        data = pd.read_csv(f"./train_data/{train_file[i+4]}", index_col="index")
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


        test_data = pd.read_csv(f"./train_data/{train_file[i+5]}", index_col="index")
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

        # 保存模型
        xgb_clf.save_model(f"./model_set/xgb-{train_file[i][:4]}-{train_file[i+4][:4]}.json")

def predict()->None:
    '''
    + 检验各个模型的预测准确率
    + 共7轮检测，即训练集和验证集在2010-2014年间的模型，用2015的个月数据进行检验，以此类推
    + 画出2015年-2021年滚动训练的7里，各模型的预测准确率折线图，保存路径为"./mons_acc.png"
    '''
    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021]:
        test_data = pd.read_csv(f"./train_data/{year}.csv", index_col="index")
        test_data_sorted = test_data.sort_values(by = "rf", ascending=False)
        long_set_test = test_data_sorted.iloc[:int(test_data_sorted.shape[0]*0.3), :]
        long_set_test["rf"] = 2
        plain_set_test = test_data_sorted.iloc[int(test_data_sorted.shape[0]*0.3):int(test_data_sorted.shape[0]*0.7), :]
        plain_set_test["rf"] = 1
        short_set_test = test_data_sorted.iloc[int(test_data_sorted.shape[0]*0.7):, :]
        short_set_test["rf"] = 0
        test_data =long_set_test.append(plain_set_test).append(short_set_test)
        X_test = test_data[features]
        y_test = test_data["rf"]

        xgb_clf = xgb.Booster()
        xgb_clf.load_model(f"./model_set/xgb-{year-5}-{year-1}.json")
        mons = [str(year) + x for x in ['-01','-02','-03','-04','-05','-06','-07','-08','-09','-10','-11','-12']]
        mons_predict = test_data.groupby("time")
        for mon in mons:
            mon_data = mons_predict.get_group(mon)
            X_test = mon_data[features]
            y_test = mon_data["rf"]
            dtest = xgb.DMatrix(X_test, feature_names=features)
            y_pred = xgb_clf.predict(dtest)
            monthly_acc[list(mon_data["time"])[0]] = round(accuracy_score(y_test,y_pred), 3)
        # 绘制预测准确率折线图
        dates = list(monthly_acc.keys())
        acc = list(monthly_acc.values())
        plt.figure( figsize=(30,10))
        plt.ylim(0.2,0.65)
        plt.xticks(fontsize=10, rotation = 90)
        plt.plot(dates, acc, marker='o', mec='r', mfc='w',label='uniprot90_train')
        plt.savefig("./mons_acc.png")

# == 回测用到的函数 ==
def generate_signal(mon_date, clf)->Tuple(list, list):
    '''
    + 传入之前训练好的模型,在每月末(最后一个交易日)依据个股的features预测其下个月的超额收益率水平: rf_Rank从高到低依次为 2(超额收益率最高的30%), 1(其他), 0(超额收益率最低的30%)
    + 选出rf_rank==2, 即超额收益最高的30%支股票, 并按市值大小进行排序, 在大市值股票和小市值股票中各选出portfolio_capacity支, 返回这些股票的num_id
    '''
    backtest_data = pd.read_csv(f"../stock_data/train_data/{mon_date[:4]}.csv")
    backtest_data = backtest_data[backtest_data["time"] == mon_date]
    X_backtest = backtest_data[features]
    Dbacktest = xgb.DMatrix(X_backtest, feature_names=features)
    y_pred = clf.predict(Dbacktest)
    backtest_data["predict_rank"] = list(y_pred)
    largeCap_long = list(backtest_data[backtest_data["predict_rank"]==2].sort_values(by = "Capitalization")[:portfolio_capacity]["num_id"])
    smallCap_long = list(backtest_data[backtest_data["predict_rank"]==2].sort_values(by = "Capitalization")[-portfolio_capacity:]["num_id"])
    return largeCap_long, smallCap_long

def Short(short_date, largeCap_long, smallCap_long, remain_Cash)->Sequence(list, list, float):
    '''
    + 根据调仓信号，先进行平仓
    + 对于持仓中不在generate_signal函数给出的largeCap_long, smallCap_long的个股, 在月初(第一个交易日)全部平仓, 对于在其中的个股, 则继续持仓
    + 平仓后, 返回仍需要建仓的股票num_id, 以及平仓后的账户现金
    '''
    #平仓
    largeCap_long_removeList = []
    largeCap_portfolio_removeList = []
    for item in largeCap_portfolio: # 遍历largeCap持仓
        flag = 0
        for stock_to_long in largeCap_long:
            if item == str(stock_to_long): # 如果想要加仓的个股已经在portfolio内，则继续持有
                largeCap_long_removeList.append(stock_to_long)
                flag = 1
                break
        if not flag: # 如果现持有的某个股不在此次调仓的加仓信号中，则全部卖出
            revenue = largeCap_portfolio[item][0]*price.loc[stock_id[item], start_map[short_date]]["Open"]*100*(1-tax_rate)
            largeCap_portfolio_removeList.append(item)
            remain_Cash += round(revenue, 3)
    for llr in largeCap_long_removeList:
        largeCap_long.remove(llr)
    for lpr in largeCap_portfolio_removeList:
        largeCap_portfolio.pop(lpr)

    smallCap_long_removeList = []
    smallCap_portfolio_removeList = []
    for item in smallCap_portfolio: # 遍历smallCap持仓
        flag = 0
        for stock_to_long in smallCap_long:
            if item == str(stock_to_long): # 如果想要加仓的个股已经在portfolio内，则继续持有
                smallCap_long_removeList.append(stock_to_long)
                flag = 1
                break
        if not flag: # 如果现持有的某个股不在此次调仓的加仓信号中，则全部卖出
            revenue = smallCap_portfolio[item][0]*price.loc[stock_id[item], start_map[short_date]]["Open"]*100*(1-tax_rate)
            smallCap_portfolio_removeList.append(item)
            remain_Cash += round(revenue, 3)
    for slr in smallCap_long_removeList:
        smallCap_long.remove(slr)
    for spr in smallCap_portfolio_removeList:
        smallCap_portfolio.pop(spr)

    return largeCap_long, smallCap_long, remain_Cash

def Long(long_date, smallCap_long, largeCap_long, remain_Cash)->float:
    '''
    + 平仓后进行建仓, 买入Short函数返回的需要建仓的个股(设其中小市值股票m支, 大市值股票n支)
    + 建仓时, 将账户现金的(1-Cap_leverage)部分均分到m个小市值个股上. 建仓时为了规避冲击成本, 成交量应不大于当日该个股总成交量的vol_rate倍, 此外交易时还应考虑交易成本tax_rate
    + 小市值股票建仓完毕后, 将账户中剩下的现金均分到n个大市值股票上, 同样限制成交量且考虑交易成本
    + 由于之前的操作中没有判断这些个股是否能交易, 因此存在调仓日某支个股price==nan的情况(即无法交易), 因此使用try:... except:... 来应对
    + 部分个股在调仓日无法交易导致Long函数结束后账户现金并不为0
    '''
    # 买入rank为2的大市值公司和小市值公司各15股，不在这个范围之内的则平仓
    # 除非受val_rate的限制，否则分配在各大市值公司上的cash相同，分配在各小市值公司上的cash也相同
    Cash_for_smallCap = (remain_Cash * (1 - Cap_leverage))/len(smallCap_long)

    for stock in smallCap_long:
        # 交易单位为手（100股）
        try:
            long_vol = min(int(Cash_for_smallCap/price.loc[stock_id[str(stock)], start_map[long_date]]["Open"]*0.01), int(price.loc[stock_id[str(stock)], start_map[long_date]]["Volume"]*vol_rate))
            # long_vol: 需要买多少股（本地数据中volumes单位为手）
            if long_vol > 0:
                cost = round(price.loc[stock_id[str(stock)], start_map[long_date]]["Open"]*long_vol*100*(1+tax_rate), 3)
                remain_Cash -= cost
                smallCap_portfolio[str(stock)] = (long_vol, cost) 
        except: # 可能有的个股在调仓日没有交易，price为nan
            continue

    Cash_for_largeCap = remain_Cash /len(largeCap_long)
    for stock in largeCap_long:
        try:
            long_vol = min(int(Cash_for_largeCap /price.loc[stock_id[str(stock)], start_map[long_date]]["Open"]*0.01), int(price.loc[stock_id[str(stock)], start_map[long_date]]["Volume"]*vol_rate))
            # long_vol: 需要买多少股（本地数据中volumes单位为手）
            if long_vol > 0:
                cost = round(price.loc[stock_id[str(stock)], start_map[long_date]]["Open"]*long_vol*100*(1+tax_rate), 3)
                remain_Cash -= cost
                largeCap_portfolio[str(stock)] = (long_vol, cost) 
        except: # 可能有的个股在调仓日没有交易，price为nan
            continue

    return remain_Cash

def Accounting(mon_date)->None:
    '''
    + 计算每个月月末账户价值, 包括: 大市值portfolio的价值(持仓量*月末交易日的收盘价), 小市值portfolio的价值(持仓量*月末交易日的收盘价), 账户总价值(前两项+剩余现金)
    + 将这些数值存入全局对象largeCap_Account, smallCap_Account, total_Account中, 以便绘图和观测
    '''
    largeCap_value = 0
    smallCap_value = 0
    for item in largeCap_portfolio:
        largeCap_value += largeCap_portfolio[item][0]*price.loc[stock_id[item], end_map[mon_date]]["Close"]*100
    for item in smallCap_portfolio:
        smallCap_value += smallCap_portfolio[item][0]*price.loc[stock_id[item], end_map[mon_date]]["Close"]*100
    largeCap_Account[mon_date] = round(largeCap_value,3)
    smallCap_Account[mon_date] = round(smallCap_value,3)
    total_Account[mon_date] = round(largeCap_value + smallCap_value + Cash, 3)

def backtest()->None:
    '''
    + 对2015年-2021年共7年数据进行回测：
    + 1. 用每个年份相对应的训练时期的模型来预测该年各月份超额回报率水平分布
    + 2. 选择rf_Rank最大的前30%支股票，按市值(capitalization)大小进行排序，分别选择一部分大市值和小市值股票
    + 3. 先平仓，将不再持仓中不在2中范围内的股票全部卖出
    + 4. 然后建仓，买入2中范围内的股票
    + 5. 在交易中考虑0.0014(0.14%)的交易成本
    + 6. 为了避免冲击成本，限制每支个股的成交量不大于其当日总成交量的0.0001倍
    + 7. 将回测结果保存到路径"./backtest.png"下
    '''
    M_ = [x[:7] for x in M_start_date] 
    start_map = dict(zip(M_, M_start_date))
    end_map = dict(zip(M_, M_end_date))
    # 加载之前训练好的模型
    xgb_model = []
    for i in range(7):
        xgb_clf = xgb.Booster()
        xgb_clf.load_model(f"./model_set/xgb-{2010+i}-{2014+i}.json")
        xgb_model.append(xgb_clf)
    # 建立模型映射，方便后续操作
    model_map = {}
    for i in range(6):
        for j in range(12):
            model_map[M_[60+i*12+j]] = xgb_model[i]
    for i in range(13):
        model_map[M_[132+i]]=xgb_model[6]
    # 开始回测，回测区间为2015-2021共7年，每一年使用的xgb_clf不同（滚动训练，滚动预测）
    model_map["2014-12"] = xgb_model[0]
    Accounting(M_[60])
    for rank in range(len(M_[60:-1])):
        large_set, small_set = generate_signal(M_[60:][rank], xgb_model[0])
        large_set, small_set, Cash = Short(M_[60:][rank+1], large_set, small_set, Cash)
        Cash = Long(M_[60:][rank+1], large_set, small_set, Cash)
        Accounting(M_[60:][rank+1])
    # 以沪深300为baseline，其仓位控制为之增不平
    hs300_Account_baseline = 10000000
    hs300_total_Account = {}
    hs300_total_vol = 0
    for date in M_[60:]:
        max_vol = int(hs300_Account_baseline / list(hs300[hs300["Unnamed: 0"]==start_map[date]]["open"])[0])
        hs300_total_vol += max_vol
        hs300_Account_baseline -= max_vol * list(hs300[hs300["Unnamed: 0"]==start_map[date]]["open"])[0]
        hs300_total_Account[date] = round(hs300_Account_baseline + hs300_total_vol*list(hs300[hs300["Unnamed: 0"]==end_map[date]]["close"])[0], 3)
    # 绘制出回测过程中各指标
    x_date = M_[60:]
    y_large_account = largeCap_Account.values()
    y_small_account = smallCap_Account.values()
    y_total_account = total_Account.values()
    y_baseline = hs300_total_Account.values()
    plt.figure(figsize=(40, 10))
    plt.xticks(rotation=45)
    plt.plot(x_date, y_large_account, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x_date, y_small_account, marker='o', markersize=3)
    plt.plot(x_date, y_total_account, marker='o', markersize=3)
    plt.plot(x_date, y_baseline, marker='o', markersize=3)
    plt.legend([ 'value of large-portfolio','value of small-portfolio', 'Total_Account', 'Baseline-hs300'])  # 设置折线名称
    plt.savefig("./backtest.png")


if __name__ == "__main__":
    # 基于本地数据生成训练样本
    data_generate()
    # 训练样本预处理
    data_handler()
    # 模型r滚动训练
    train_xgb_model()
    # 模型滚动预测
    predict()
    # 分时段(7段)滚动回测
    backtest()

