import pandas as pd
data_set = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021+"]
for data_time in data_set:
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


