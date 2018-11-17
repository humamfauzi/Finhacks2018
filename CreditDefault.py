import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

import warnings
warnings.filterwarnings('ignore')

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
            if dataframe[i].nunique() < 28:
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

print("Data: ",os.listdir("Credit Score/data_input"))

df_train = pd.read_csv("Credit Score/data_input/npl_train.csv")
df_test = pd.read_csv("Credit Score/data_input/npl_test.csv")

print("Train Shape: ",df_train.shape)
print("Test Shape: ",df_test.shape)

print("Test to train ratio: ", df_test.shape[0]/df_train.shape[0])

target = df_train["flag_kredit_macet"]
train_id = df_train["X"]
test_id = df_test["X"]
print("Target number:",dict(target.value_counts()))
df_train.drop("X", axis=1, inplace=True)
df_test.drop("X", axis=1, inplace=True)
df_train["kode_cabang"] = df_train["kode_cabang"].fillna("X")
df_test["kode_cabang"] = df_test["kode_cabang"].fillna("X")

# Kredit lunas
condition_0 = df_train["flag_kredit_macet"] == 0

# Kredit macet
condition_1 = df_train["flag_kredit_macet"] == 1 

# Describing Minimum Accuracy
a = len(target[condition_0])
b = len(target[condition_1])

minimum_accuracy = a / (a + b)
print("Minimum Accuracy:", minimum_accuracy)

default_ratio_overall = 1 - minimum_accuracy
print("Overall Default Ratio:", default_ratio_overall)

train_desc = describe_dataframe(df_train)

disc_col, cont_col = [], []
for i in train_desc.keys():
    for j in train_desc[i].keys():
        disc_col.append(j)
        
for i in df_test.columns:
    if i not in disc_col:
        cont_col.append(i)

print("\n Discrete Column")
for num, col in enumerate(disc_col):
    print(str(num) + ".",col)

print("\n Continuous Column")
for num, col in enumerate(cont_col):
    print(str(num) + ".",col)

# Describing continuous column 
q0 = df_train[cont_col].describe().T
q0.drop("count", axis=1, inplace=True)

q0["skew"] = [df_train[i].skew() for i in q0.index]
q0["kurt"] = [df_train[i].kurt() for i in q0.index]
q0["range"] = q0["max"] - q0["min"]

q1 = df_test[cont_col].describe().T
q1.drop("count", axis=1, inplace=True)

q1["skew"] = [df_test[i].skew() for i in q1.index]
q1["kurt"] = [df_test[i].kurt() for i in q1.index]
q1["range"] = q1["max"] - q1["min"]
q1.index = [i + "_test" for i in q1.index]

data_detail = pd.concat([q0,q1])

# Payment inactivity

a = df_train["rasio_pembayaran"] < 10
b = df_train["rasio_pembayaran_3bulan"] < 10
c = df_train["rasio_pembayaran_6bulan"] < 10

a_t = df_test["rasio_pembayaran"] < 10
b_t = df_test["rasio_pembayaran_3bulan"] < 10
c_t = df_test["rasio_pembayaran_6bulan"] < 10

df_train["tidak_mampu_bayar"] = a & b & c
df_test["tidak_mampu_bayar"] = a_t & b_t & c_t

q = df_train["tidak_mampu_bayar"]
df_train["tidak_mampu_bayar"] = q.replace(q.unique(), range(q.nunique()))

q = df_test["tidak_mampu_bayar"]
df_test["tidak_mampu_bayar"] = q.replace(q.unique(), range(q.nunique()))

# Usage in Number

df_train["pemakaian_3bln"] = df_train["limit_kredit"] * df_train["utilisasi_3bulan"]
df_train["pemakaian_6bln"] = df_train["limit_kredit"] * df_train["utilisasi_6bulan"]

df_test["pemakaian_3bln"] = df_test["limit_kredit"] * df_test["utilisasi_3bulan"]
df_test["pemakaian_6bln"] = df_test["limit_kredit"] * df_test["utilisasi_6bulan"]

# Usage inactivity

a = df_train["total_pemakaian_per_limit"] ==0
b = df_train["pemakaian_3bln_per_limit"] ==0
c = df_train["pemakaian_6bln_per_limit"] ==0

df_train["tidak_aktif"] = a & b & c

a = df_test["total_pemakaian_per_limit"] == 0
b = df_test["pemakaian_3bln_per_limit"] == 0
c = df_test["pemakaian_6bln_per_limit"] == 0

df_test["tidak_aktif"] = a & b & c

# Digitize tahun pembukaan
q = np.percentile(df_train.jumlah_tahun_sejak_pembukaan_kredit, np.linspace(0,100, 10))
q = pd.cut(df_train.jumlah_tahun_sejak_pembukaan_kredit, q, include_lowest=True)
q = q.replace(q.unique(), range(q.nunique()))

df_train.jumlah_tahun_sejak_pembukaan_kredit = q

q = np.percentile(df_test.jumlah_tahun_sejak_pembukaan_kredit, np.linspace(0,100,10))
q = pd.cut(df_test.jumlah_tahun_sejak_pembukaan_kredit, q, include_lowest=True)
q = q.replace(q.unique(), range(q.nunique()))

df_test.jumlah_tahun_sejak_pembukaan_kredit = q


def isAble2Transform(Series):
    condition = len(Series[Series <= 0]) == 0 
    condition_0 = len(Series[Series < 1][Series > -1]) == 0
    condition_1 = len(Series[Series > 0][Series < 1][Series != 0]) == 0
    if condition:
        print(str(Series.name) + " can be transformed!")
        return Series.name
    elif condition_0:
        print(str(Series.name) + " can be transformed!")
        return Series.name
    elif condition_1:
        print(str(Series.name) + " can be transformed!")
        return Series.name
    else:
        print("There is a value between [-1, 1] --- " + str(Series.name) + " advised not to be transformed")
        return None
        
print("\n")

transform_column = []
for i in q0.index:
    transform_column.append(isAble2Transform(df_train[i]))

# List of value that can be transformed by log transform
transform_column = [i for i in transform_column if i is not None]

unable_trans = [i for i in q0.index if i not in transform_column]

print("\nFeature that unable to transform but have a high skewness:")
for i in data_detail[np.abs(data_detail["skew"]) > 5].index:
    if i in unable_trans:
        print("-",i)

over_kurt = data_detail[np.abs(data_detail["kurt"]) > 10].index
print("\nFeature that unable to transform but have a high kurtosis:")
for i in over_kurt:
    if i in unable_trans:
        print("-",i)

# Transforming column with log10 transform
for i in transform_column:
    df_train[i] = zscore(log10_transform(df_train[i]))
    df_test[i] = zscore(log10_transform(df_test[i]))

# Overlimit handling; later will be numbered since high > medium > low > no relationship will be retained
# Need to be transformed into a function for better readibility
Series = df_train["persentasi_overlimit"]
percentil = np.percentile(Series, np.linspace(0,100,11))
binned = pd.cut(Series, np.unique(percentil), include_lowest=True)

Series_test = df_test["persentasi_overlimit"]
percentil_test = np.percentile(Series_test, np.linspace(0,100,11))
binned_test = pd.cut(Series_test, np.unique(percentil_test), include_lowest=True)

index__ = binned.value_counts().sort_index().index
index__test = binned_test.value_counts().sort_index().index
replacement = range(len(index__))
replacement_test = range(len(index__test))

binned = binned.replace(index__, replacement)
binned_test = binned_test.replace(index__test, replacement_test)

df_train[Series.name] = binned
df_test[Series.name] = binned_test

# rasio_pembayaran = bayaran/tagihan; also applies to its derivative
Series = df_train["rasio_pembayaran"]
percentil = np.percentile(Series, np.linspace(0,100,11))
binned = pd.cut(Series, np.unique(percentil), include_lowest=True)

Series_test = df_test["rasio_pembayaran"]
percentil_test = np.percentile(Series_test, np.linspace(0,100,11))
binned_test = pd.cut(Series_test, np.unique(percentil_test), include_lowest=True)

index__ = binned.value_counts().sort_index().index
index__test = binned_test.value_counts().sort_index().index
replacement = range(len(index__))
replacement_test = range(len(index__test))

binned = binned.replace(index__, replacement)
binned_test = binned_test.replace(index__test, replacement_test)

df_train[Series.name] = binned
df_test[Series.name] = binned_test

Series = df_train["rasio_pembayaran_3bulan"]
percentil = np.percentile(Series, np.linspace(0,100,11))
binned = pd.cut(df_train.rasio_pembayaran_3bulan, np.unique(percentil), include_lowest=True)

Series_test = df_test["rasio_pembayaran_3bulan"]
percentil_test = np.percentile(Series_test, np.linspace(0,100,11))
binned_test = pd.cut(Series_test, np.unique(percentil_test), include_lowest=True)

index__ = binned.value_counts().sort_index().index
index__test = binned_test.value_counts().sort_index().index
replacement = range(len(index__))
replacement_test = range(len(index__test))

binned = binned.replace(index__, replacement)
binned_test = binned_test.replace(index__test, replacement_test)

df_train[Series.name] = binned
df_test[Series.name] = binned_test

# Get dummies for discrete variable
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

print("After Dummies train shape:",df_train.shape)
print("After Dummies test shape:",df_test.shape)

for i in df_train.columns:
    if i not in df_test.columns:
        print("NOT IN TEST: ",i)
        
for i in df_test.columns:
    if i not in df_train.columns:
        print("NOT IN TRAIN:",i)

# --------------------------------------- 
#                MODELLING 
# --------------------------------------- 

print("START MODELLING...")

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

X, y = df_train.drop("flag_kredit_macet", axis=1), df_train["flag_kredit_macet"]
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
estimators.append(SGDClassifier(random_state=rstate, loss="log"))
estimators.append(LogisticRegression(random_state=rstate))
estimators.append(DecisionTreeClassifier(random_state=rstate, max_depth=5))
estimators.append(GradientBoostingClassifier(random_state=rstate))
estimators.append(LinearDiscriminantAnalysis())

estimator_list = ["SGD", "LR", "DTC", "GBC", "LDA"]
est_mean, est_std = [], []
for i, estimator in enumerate(estimators):
    print("Fitting", estimator_list[i])
    arr, mean, std = manual_cross_validation(estimator, X, y)
    est_mean.append(mean)
    est_std.append(std)

# For Visualization
# cv_res = pd.DataFrame({"CV_AVG": est_mean, "CV_STD": est_std, "Estimator": estimator_list})
# g = sns.barplot("CV_AVG", "Estimator", data=cv_res, orient="h")
# g.set_xlabel("AVERAGE SCORE")
# g = g.set_title("CROSS VALIDATION SCORE")

i = pd.DataFrame([est_mean, est_std]).T
i.index=estimator_list
i.columns=["mean", "std"]
print(i)

# Selected Estimator
LR_params = {
    "penalty": ["l2"],
    "tol": [1e-6],
    "C": [1e-2],
    "solver": ["liblinear", "newton-cg", "sag"],
    "max_iter": [4e3, 6e3],
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
    "n_estimators": [7000, 7500],
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
                      verbose=1, n_jobs=4)
GBC_GS.fit(X, y)
GBC_best = GBC_GS.best_estimator_
print(GBC_GS.best_params_)

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

print("GBC Final Score on Rigid Prediction:", GBC_final_score)

GBC_final_predict = pd.Series(GBC_GS.best_estimator_.predict_proba(df_test)[:,1])
test_result = probs_threshold(GBC_final_predict, GBC_threshold)
result_df = pd.concat([test_id, pd.Series(test_result), GBC_final_predict], axis=1)
result_df.to_csv('CS_challenge.csv', index=False)
