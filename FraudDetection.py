import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

import warnings
warnings.filterwarnings('ignore')

import pprint

from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline

def describe_dataframe(dataframe):
    # Create a dictionary which have every detected categorical value counts in a dataframe and NaN ratio 
    describe_df = {}
    for j in dataframe.dtypes.value_counts().index:
        describe_dict = {}
        cols = dataframe.select_dtypes([j]).columns
        for i in cols:
            if dataframe[i].nunique() < 100:
                describe_dict[str(i)] = {"total_category": dict(dataframe[i].value_counts()), 
                                         "NaN_ratio": dataframe[i].isnull().sum()/float(dataframe.shape[0])}
        describe_df[str(j)] = describe_dict
        print('Total', str(j), 'Classification:', len(describe_dict.keys()), "from", len(cols))
    return describe_df

def log10_transform(ret_ser):
    Series = pd.Series(ret_ser)
    positive = Series[Series > 1]
    negative = Series[Series < -1]
    compressed = Series[Series < 1][Series > -1]
    Series.loc[positive.index] = np.log10(positive)
    Series.loc[negative.index] = np.log10(abs(negative)) * -1
    Series.loc[compressed.index] = 0
    return Series

def evaluate_members(Series, condition_0, condtion_1, overall_target_ratio):
    a = Series.value_counts()
    b = Series[condition_0].value_counts()
    c = Series[condition_1].value_counts()
    conclusion = pd.concat([a, b, c], axis=1)
    conclusion.columns = ["Total", "0", "1"]
    
    conclusion["Ratio_0"] = b / a
    conclusion["Ratio_1"] = c / a
    conclusion["HigherThanOverall"] = conclusion["Ratio_1"].apply(lambda x: True if x > overall_target_ratio else False)
    conclusion.fillna(value=0, inplace=True)
    return conclusion

print(os.listdir("Fraud/data_input"))

df_train = pd.read_csv("Fraud/data_input/fraud_train.csv")
df_test = pd.read_csv("Fraud/data_input/fraud_test.csv")
for i in ["rata_rata_nilai_transaksi", "maksimum_nilai_transaksi", "minimum_nilai_transaksi", "rata_rata_jumlah_transaksi"]:
    df_train[i].fillna(np.mean(df_train[i]), inplace=True)
    df_test[i].fillna(np.mean(df_test[i]), inplace=True)

print("Train Shape: ",df_train.shape)
print("Test Shape: ",df_test.shape)

print("Test to train ratio: ", df_test.shape[0]/df_train.shape[0])

target = df_train["flag_transaksi_fraud"]
train_id = df_train["X"]
test_id = df_test["X"]
print("Target number:",dict(target.value_counts()))
df_train.drop("X", axis=1, inplace=True)
df_test.drop("X", axis=1, inplace=True)

# Kredit lunas
condition_0 = df_train["flag_transaksi_fraud"] == 0

# Kredit macet
condition_1 = df_train["flag_transaksi_fraud"] == 1 

# Describing Minimum Accuracy
a = len(target[condition_0])
b = len(target[condition_1])

minimum_accuracy = b / (a + b)
print("Minimum Accuracy:",minimum_accuracy)

# Separating discrete and continuous column and printing the result
train_desc = describe_dataframe(df_train)

disc_col, cont_col = [], []
for i in train_desc.keys():
    for j in train_desc[i].keys():
        disc_col.append(j)
        
for i in df_test.columns:
    if i not in disc_col:
        cont_col.append(i)

del cont_col[0]
print("\nDiscrete Column")

print(pd.DataFrame([df_train[i].nunique() for i in disc_col], index=disc_col, columns=["Unique_value"]))

print("\nContinuous Column")

print(pd.DataFrame([df_train[i].nunique() for i in cont_col], index=cont_col, columns=["Unique_value"]))

print("Unique customer in Train: ", train_id.nunique())
print("Unique customer in Test: ", test_id.nunique())

def tanggal_conversion(number):
    if number < 31:
        return "Januari"
    elif number >= 31 and number < 59:
        return "Feburari"
    elif number >= 59 and number < 90:
        return "Maret"
    elif number >= 90 and number < 120:
        return "April"
    elif number >= 120 and number < 151:
        return "Mei"
    elif number >= 151 and number < 181:
        return "Juni"
    elif number >= 181 and number < 212:
        return "Juli"
    elif number >= 212 and number < 242:
        return "Agustus"
    elif number >= 242 and number < 273:
        return "September"
    elif number >= 273 and number < 303:
        return "Oktober"
    elif number >= 303 and number < 333:
        return "November"
    elif number >= 333:
        return "Desember"

def week_conversion(number):
    if number < 8:
        return 'Week 1'
    elif number > 8 and number < 16:
        return 'Week 2'
    elif number > 16 and number < 24:
        return 'Week 3'
    elif number > 24:
        return 'Week 4'

def check_data_existance(df_train, df_test, column):
    not_exist_in_test = []
    not_exist_in_train = []
    for i in df_train[column].unique():
        if i not in df_test[column].unique():
            not_exist_in_test.append(i)
            
    for k in df_test[column].unique():
        if k not in df_train[column].unique():
            not_exist_in_train.append(k)
    
    return not_exist_in_train, not_exist_in_test

df_train["Tanggal"] = df_train["id_tanggal_transaksi_awal"] - df_train["id_tanggal_transaksi_awal"].min()
df_test["Tanggal"] = df_test["id_tanggal_transaksi_awal"] - df_test["id_tanggal_transaksi_awal"].min()

df_train["bulan"] = [tanggal_conversion(i) for i in df_train["Tanggal"]]
df_test["bulan"] = [tanggal_conversion(i) for i in df_test["Tanggal"]]

df_train["minggu"] = [week_conversion(i%30 + 1) for i in df_train["Tanggal"]]
df_test["minggu"] = [week_conversion(i%30 + 1) for i in df_test["Tanggal"]]

# ["id_merchant", "tipe_mesin", "nama_negara", "pemilik_mesin"]
# Dominant Merchant (mungkin ATM) dan Non-dominant merchant (mungking EDC)
df_train["new_id_merchant"] = [0 if i == -2 else 1 for i in df_train["id_merchant"]]
df_test["new_id_merchant"] = [0 if i == -2 else 1 for i in df_test["id_merchant"]]

# -3 dalam tipe mesin adalah ATM dan sisanya adalah EDC
df_train["new_tipe_mesin"] = [0 if i == -3 else 1 for i in df_train["tipe_mesin"]]
df_test["new_tipe_mesin"] = [0 if i == -3 else 1 for i in df_test["tipe_mesin"]]

# 5 adalah dominan dan sisanya adalah non-dominan
df_train["new_nama_negara"] = [0 if i == 5 else 1 for i in df_train["nama_negara"]]
df_test["new_nama_negara"] = [0 if i == 5 else 1 for i in df_test["nama_negara"]]

df_train["new_pemiliki_mesin"] = [0 if i == 613 else 1 for i in df_train["pemilik_mesin"]]
df_test["new_pemiliki_mesin"] = [0 if i == 613 else 1 for i in df_test["pemilik_mesin"]]

df_train["nilai_transaksi"] = np.log(df_train["nilai_transaksi"])
df_test["nilai_transaksi"] = np.log(df_test["nilai_transaksi"])

percentile = np.percentile(df_train["nama_kota"], np.linspace(0,100, 20))
df_train["new_nama_kota"] = pd.cut(df_train["nama_kota"], np.unique(percentile), include_lowest=True)
df_train["new_nama_kota"].replace(df_train["new_nama_kota"].unique(), list("QWERTYUIOP"), inplace=True)

df_test["new_nama_kota"] = pd.cut(df_test["nama_kota"], np.unique(percentile), include_lowest=True)
df_test["new_nama_kota"].replace(df_test["new_nama_kota"].unique(), list("QWERTYUIOP"), inplace=True)

df_train.minimum_nilai_transaksi = np.log(df_train.minimum_nilai_transaksi)
df_test.minimum_nilai_transaksi = np.log(df_test.minimum_nilai_transaksi)

df_train.rata_rata_nilai_transaksi = np.log(df_train.rata_rata_nilai_transaksi)
df_test.rata_rata_nilai_transaksi = np.log(df_test.rata_rata_nilai_transaksi)

df_train["nilai_transaksi <> minimum_nilai"] = df_train["nilai_transaksi"] * df_train["minimum_nilai_transaksi"]
df_test["nilai_transaksi <> minimum_nilai"] = df_test["nilai_transaksi"] * df_test["minimum_nilai_transaksi"]

# Vanilla Column
similar_columns = []
df_train.drop(["status_transaksi", "bank_pemilik_kartu","flag_transaksi_finansial"] + similar_columns, axis=1, inplace=True)
df_test.drop(["status_transaksi", "bank_pemilik_kartu","flag_transaksi_finansial"] + similar_columns, axis=1, inplace=True)

df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

# Checking column fitness
for i in df_train.columns:
    if i not in df_test.columns:
        print("NOT IN TEST: ",i)
        
for i in df_test.columns:
    if i not in df_train.columns:
        print("NOT IN TRAIN:",i)
        
print(df_train.shape)
print(df_test.shape)

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = df_train.drop("flag_transaksi_fraud", axis=1), df_train["flag_transaksi_fraud"]
rstate = 77
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rstate)

def manual_cross_validation(estimator, X, y, cv=StratifiedKFold(n_splits = 5)):
    arr = []
    for train_index, test_index in cv.split(X, y):
        estimator.fit(X.loc[train_index], y.loc[train_index])
        estimator_probs = estimator.predict_proba(X.loc[test_index])
        arr.append(roc_auc_score(y.loc[test_index], estimator_probs[:,1]))
    return arr, np.mean(arr), np.std(arr)

estimators = []
estimators.append(LogisticRegression(random_state=rstate))
estimators.append(DecisionTreeClassifier(random_state=rstate, max_depth=5))
estimators.append(GradientBoostingClassifier(random_state=rstate))
estimators.append(LinearDiscriminantAnalysis())

estimator_list = ["LR", "DTC", "GBC", "LDA"]
est_mean, est_std = [], []
for i, estimator in enumerate(estimators):
    print("Fitting", estimator_list[i])
    arr, mean, std = manual_cross_validation(estimator, X, y)
    est_mean.append(mean)
    est_std.append(std)

# For visualization
# cv_res = pd.DataFrame({"CV_AVG": est_mean, "CV_STD": est_std, "Estimator": estimator_list})
# g = sns.barplot("CV_AVG", "Estimator", data=cv_res, orient="h")
# g.set_xlabel("AVERAGE SCORE")
# g = g.set_title("CROSS VALIDATION SCORE")
print(pd.DataFrame([est_mean, est_std], index=estimator_list, columns=["mean", "std"]))

LR_params = {
    "penalty": ["l2"],
    "tol": [1e-6],
    "C": [1e-2, 1e2],
    "solver": ["liblinear", "newton-cg", "sag"],
    "max_iter": [4e3],
    "random_state": [rstate]
}

LR_GS = GridSearchCV(LogisticRegression(), 
                     param_grid=LR_params, 
                     cv=StratifiedKFold(n_splits = 5), 
                     scoring="roc_auc", 
                     verbose=1, n_jobs=4)
LR_GS.fit(X, y)
LR_best = LR_GS.best_estimator_
print(LR_GS.best_params_)

DTC_params = {
    "splitter": ["best", "random"],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 3, 5],
    "max_features": [0.1, 0.5, 0.8],
    "max_leaf_nodes": [3, 5, None],
    "random_state": [rstate]
}

DTC_GS = GridSearchCV(DecisionTreeClassifier(), 
                      param_grid=DTC_params, 
                      cv=StratifiedKFold(n_splits = 5), 
                      scoring="roc_auc", 
                      verbose=1, n_jobs=4)
DTC_GS.fit(X, y)
DTC_best = DTC_GS.best_estimator_
print(DTC_GS.best_params_)
GBC_params = {
    "learning_rate": [1e-3],
    "n_estimators": [4000, 6000, 8000],
    "max_depth": [5],
    "subsample": [0.8],
    "max_features": [None],
    "init": [None],
    "random_state": [rstate]
}

GBC_GS = GridSearchCV(GradientBoostingClassifier(), 
                      param_grid=GBC_params, 
                      cv=StratifiedKFold(n_splits = 5), 
                      scoring="roc_auc", 
                      verbose=2, n_jobs=4)
GBC_GS.fit(X, y)
GBC_best = GBC_GS.best_estimator_
print(GBC_GS.best_params_)

print("GBC Score: ", GBC_GS.best_score_)
print("LR Score:", LR_GS.best_score_)
print("DTC Score:", DTC_GS.best_score_)

def probs_threshold(array, threshold):
    return [1 if i >= threshold else 0 for i in array]

def BestThreshold(estimator, X, y, range_prob=np.linspace(1e-4, 0.2, 1000)):
    estimator_predict = estimator.best_estimator_.predict_proba(X)
    roc_collection = []
    index_dict = {}
    for i in range_prob:
        rigid = probs_threshold(estimator_predict[:,1], i)
        roc = roc_auc_score(y, rigid)
        roc_collection.append(roc)
        index_dict[roc] = i
    return index_dict[max(roc_collection)]

GBC_threshold = BestThreshold(GBC_GS, X_test, y_test)
GBC_rigid_final = probs_threshold(GBC_GS.best_estimator_.predict_proba(X_test)[:,1], GBC_threshold)
GBC_final_score = roc_auc_score(y_test, GBC_rigid_final)

print("GBC Best Score For Rigid Prediction: ", GBC_final_score)

GBC_final_predict = pd.Series(GBC_GS.best_estimator_.predict_proba(df_test)[:,1])
test_result = probs_threshold(GBC_final_predict, GBC_threshold)
result_df = pd.concat([test_id, pd.Series(test_result), GBC_final_predict], axis=1)
result_df.to_csv('fraud_challenge.csv', index=False)
