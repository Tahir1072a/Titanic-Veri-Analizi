#%% md
# # Introduction
# 
# [**Content**](#):
# 1. Load and Check Data
# 2. Variable Description
#    - Univariate Variable Analysis  => Her bir değişkenin tek başına incelenmesi...
#        - Categorical Variables
#        - Numerical Variables
# 3. Basic Data Analysis
# 4. Outlier Detection
# 5. Missing Value
#    - Find Missing Value
#    - Fill Missing Value
# 6. Visualization
#     - Correlation between Sibsp - Parch - Age - Fare - Survived
#     - Sibsp -- Survived
#     - PClass -- Survived
#     - Age -- Survived
#     - Pclass -- Survived -- Age
#     - Embarked -- Sex -- Fare -- Survived
#     - Fill Missing, Age Feature:
# 7. Feature Engineering
# 8. Modeling
#    - Train Test Split
#    - Simple Logistic Regression
#    - Hyperparameter Tunning - Grid Search - Cross Validation
#    - Prediction and Submission
#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import os

# Dosyaların tam yollarını almak için kullanılır
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#%% md
# ## Load And Check Data
#%%
train_df = pd.read_csv("../data/train.csv")

X_test = pd.read_csv("../data/test.csv")
y_test = pd.read_csv("../data/gender_submission.csv")

X_test["Survived"] = y_test["Survived"]
train_df = pd.concat([train_df, X_test])
#%%
train_df.columns
#%%
train_df.head()
#%%
train_df.describe()
#%% md
# # Variable Description
#%%
train_df.info()
#%% md
# ## Univariate Variable Analysis
# 
# - Categorical: Survived, Pclass, Embarked, Cabin, SibSp, Parch, Name, Sex, Ticket
# - Numeric Variables: Fare, age, PassengerId
# 
#%% md
# ### Categorical Variables
#%%
def bar_plot(variable):
    
    # get feature
    var = train_df[variable]
    varValue = var.value_counts()

    # visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(ticks=varValue.index, labels=varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print(f"{variable}: \n {varValue}")
#%%
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)
#%%
category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print(f"{train_df[c].value_counts()}\n ")
#%% md
# ### Numerical Variable
#%%
def plot_hist(variable):

    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"{variable} distribution with hist")
    plt.show()
#%%
numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)
#%% md
# # Basic Data Analysis
#%%
# Pclass vs Survived
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
#%%
# Sex - Survived
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
#%%
# SibSp vs Survived
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
#%%
# Parch vs Survived
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
#%% md
# # Outlier Detection
#%%
def detect_outliers(df, features):
    outlier_indices = []

    for c in features:
        Q1 = np.percentile(df[c], 25)
        Q3 = np.percentile(df[c], 75)

        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
#%%
train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
#%%
# Drop Outliers
train_df = train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(drop=True)
#%% md
# # Missing Value
#%%
train_df_len = len(train_df)
#%% md
# ## Find Missing Values
#%%
train_df.columns[train_df.isnull().any()]
#%%
train_df.isnull().sum()
#%% md
# ## Fill Missing Values
# - Embarked has 2 missing value
#%%
train_df[train_df["Embarked"].isnull()]
#%%
train_df.boxplot(column="Fare", by="Embarked")
plt.show()
#%%
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
#%% md
# # Visualization
# 
# Featurelar arasındaki korelasyonu inceleme...
#%% md
# ### Correlation Between SibSp - Parch - Age - Fare - Survived
#%%
list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot=True, fmt=".2f")
plt.show()
#%% md
# ## Sibsp -- Survived
#%%
g = sns.catplot(x= "SibSp", y = "Survived", data= train_df, kind="bar")
g.set_ylabels("Survived Probability")
plt.show()
#%% md
# ## PClass -- Survived
#%%
g = sns.catplot(x="Pclass", y="Survived", data=train_df, kind="bar")
g.set_ylabels("Survived Probability")
plt.show()
#%% md
# ## Age -- Survived
#%%
g = sns.FacetGrid(train_df, col="Survived")
g.map(sns.distplot, "Age", bins=25)
plt.show()
#%% md
# ## Pclass -- Survived -- Age
#%%
g = sns.FacetGrid(train_df, col="Survived", row="Pclass", hue="Sex")
g.map(plt.hist, "Age", bins=25)
g.add_legend()
plt.show()
#%% md
# ## Embarked -- Sex -- Fare -- Survived
#%%
g = sns.FacetGrid(train_df, row="Embarked", col="Survived")
g.map(sns.barplot, "Sex", "Fare")
plt.show()
#%% md
# ## Fill Missing, Age Feature: 
#%%
train_df[train_df["Age"].isnull()]
#%%
sns.catplot(x="Sex", y="Age",data=train_df, kind="box")
plt.show()
#%%
sns.catplot(x="Sex", y= "Age", hue="Pclass", data=train_df, kind="box")
plt.show()
#%%
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    # Yaşı kayıp olan kişi ile SibSp Parch ve Pclass değerleri aynı olan kişileri getir ve medyan değerini al.
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & 
                                (train_df["Parch"] == train_df.iloc[i]["Parch"]) & 
                                (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_median = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_median
#%%
train_df[train_df["Age"].isnull()]
#%% md
# # Feature Engineering
#%% md
# ### Name -- Title
#%%
# Name'lere bakarak yeni bir feature çıkarımı yapabilir miyiz onu gözlemliyoruz. Mesela burada ünvanlara göre bir feature extraction yapabiliriz.
train_df["Name"].head(10)
#%%
name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)
#%%
sns.countplot(x="Title", data=train_df)
plt.xticks(rotation=60)
plt.show()
#%%
# Convert Categorical
train_df["Title"] = train_df["Title"].replace(["Lady", "the Countess", "Jonkheer", "Capt", "Col", "Sir", "Lady", "Major", "Mme", "Dr", "Rev", "Don"], "other") # Kaldırılabilir...
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)
#%%
g = sns.countplot(x="Title", data=train_df)
plt.xticks(rotation=60)
plt.show()
#%%
g = sns.catplot(x = "Title", y = "Survived", data=train_df, kind="bar")
g.set_xticklabels(["Master", "Mrs", "Mr", "Other"])
g.set_ylabels("Survival Probability")
plt.show()
#%%
train_df.drop(labels=["Name"], axis=1,inplace=True)
#%%
train_df.info()
#%%
train_df = pd.get_dummies(train_df, columns=["Title"], dtype=int)
train_df.head()
#%% md
# ### Family Size
#%%
train_df.info()
#%%
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
#%%
train_df.info()
#%%
g = sns.catplot(x = "Fsize", y = "Survived", data=train_df, kind="bar")
g.set_ylabels("Survival")
plt.show()
#%%
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df.head()
#%%
train_df.info()
#%%
sns.countplot(x = "family_size", data=train_df)
plt.show()
#%%
train_df = pd.get_dummies(train_df, columns=["family_size"], dtype=int)
train_df.info()
#%% md
# ### Embarked
#%%
train_df["Embarked"].head()
#%%
sns.countplot(x="Embarked", data=train_df)
plt.show()
#%%
train_df = pd.get_dummies(train_df, columns=["Embarked"], dtype=int)
train_df.head()
#%%
train_df.info()
#%% md
# ### PClass
#%%
sns.countplot(x="Pclass", data=train_df)
plt.show()
#%%
train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Pclass"], dtype=int)
train_df.head()
#%% md
# ### Sex
#%%
train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"], dtype=int)
train_df.head()
#%%
# Cabin numarası olmayanları 0 ile dolduruyorum. Bu feature'ı da bir sınıf olarak kabul edeceğim.

train_df['Has_Cabin'] = np.where(train_df['Cabin'].notnull(), 1, 0)
#X_test['Has_Cabin'] = np.where(X_test['Cabin'].notnull(), 1, 0)

train_df.drop('Cabin', axis=1, inplace=True)
#X_test.drop('Cabin', axis=1, inplace=True)

train_df.info()
#X_test.info()
#%% md
# ## Fare -- Missing Value
#%%
train_df[train_df["Fare"].isnull()]
#%%
train_df.dropna(subset=["Fare"], inplace=True)
train_df[train_df["Fare"].isnull()]
#%%
# Ticket ve PassengerId'in tamamen gereksiz featurelar olduğunu düşünüyorum. Bunları modele veri olarak vermyeceğim.
train_df.drop(labels=["Ticket", "PassengerId", "Fsize"], axis=1, inplace=True)
#%%
corr_matrix = train_df.corr()
print(corr_matrix["Survived"].abs().sort_values(ascending=False))
#%% md
# # Modeling
#%% md
# ### KNN ile Model Eğitimi
#%%
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#%% md
# ## Train Test Split
#%%
train = train_df.copy()

X_train = train.drop(labels="Survived", axis=1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

print(f"X train: {len(X_train)}")
print(f"X test: {len(X_test)}")
print(f"Y train: {len(y_train)}")
print(f"Y test: {len(y_train)}")
#%% md
# ## Simple Logistic Regression
#%%
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100, 2)
acc_log_test = round(logreg.score(X_test, y_test)* 100,2)

print(f"Train accuracy score: {acc_log_train}")
print(f"Test accuracy score: {acc_log_test}")
#%% md
# ## HyperParameter Tuning -- Grid Search -- CrossValidation
# 
# We will compare 5 ml classifier and evaluate mean accuracy of each of them by stratified cross validation.
# - Decision Tree
# - SVM
# - Random Forest
# - KNN
# - Logistic Regression
#%%
X_train.info()
#%%
random_state = 42
classifiers = [DecisionTreeClassifier(random_state=random_state),
               SVC(random_state=random_state),
               RandomForestClassifier(random_state=random_state),
               LogisticRegression(random_state=random_state),
               KNeighborsClassifier()]


dt_param_grid = {"min_samples_split": range(10,100,5),
                 "max_depth": range(1, 20, 1)}

svc_param_grid = {"kernel": ["rbf"],
                  "gamma": [0.001, 0.01, 0.1, 1],
                  "C": [1, 10, 50, 100, 200, 300, 1000]}

rf_param_grid = {"max_features": [2, 5, 8, 10, 12], # Rastgele Feature Bölümleme
                 "min_samples_split": [10, 25, 50],
                 "min_samples_leaf": [5, 20, 40],
                 "bootstrap": [True],
                 "n_estimators": [50, 100, 150, 200, 250, 300],
                 "criterion": ["gini"]}

logreg_param_grid = {"C": np.logspace(-3,3,7),
                     "penalty": ["l1", "l2"],
                     "solver": ["liblinear"]}

knn_param_grid = {"n_neighbors": np.linspace(1, 19, num=10, dtype=int).tolist(),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}

classifier_param = [dt_param_grid,
                    svc_param_grid,
                    rf_param_grid,
                    logreg_param_grid,
                    knn_param_grid]
#%%
cv_result = []
best_estimators = []
for i in range(len(classifiers)):
    clf = GridSearchCV(classifiers[i], param_grid=classifier_param[i], cv=StratifiedKFold(n_splits=10), scoring="accuracy", n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
#%%

#%%
cv_results = pd.DataFrame({"Cross Validation Accuracy Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM", "RandomForestTree", "LogisticRegression", "KNN"]})
cv_results
#%%
best_estimators_df = pd.DataFrame({"Cross Validation Best Estimator Parameters": best_estimators, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestTree", "LogisticRegression", "KNN"]})
best_estimators_df
#%%
s = sns.barplot(cv_results, x="ML Models", y="Cross Validation Accuracy Means")
#%% md
# ## Ensemble Modeling
#%%
# 3 farklı classifier'ın oylama yöntemi ile karar verilmesi...
votingC = VotingClassifier(estimators=[("dt", best_estimators[0]), ("rfc", best_estimators[2]), ("lr", best_estimators[3])], voting="soft", n_jobs=-1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test), y_test))
#%%
