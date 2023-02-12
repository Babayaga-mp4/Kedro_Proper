import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sqlalchemy import create_engine
from sklearn.svm import SVC
import xgboost as xgb
import pickle
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score ,recall_score ,accuracy_score ,f1_score ,roc_curve ,auc ,confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix
import time
import mlflow

warnings.filterwarnings('ignore')
path = 'train.csv'

create_engine("sqlite:///mlflow.db" ,pool_pre_ping = True)
mlflow.set_registry_uri(
    "sqlite:///mlflow.db")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
# mlflow.create_experiment('Churn Prediction Demo')



def read_data(path):
    df = pd.read_csv(path)
    return df


def preprocessing(df):
    df.churn.replace({"yes": 1 ,"no": 0} ,inplace = True)
    dummy_df = pd.get_dummies(df)
    return dummy_df


def oversampling(dummy_df):
    y = dummy_df.churn.values
    X = dummy_df.drop('churn' ,axis = 1)
    cols = X.columns

    smt = SMOTE(sampling_strategy = 0.7)
    X ,y = smt.fit_resample(X ,y)
    return X ,y ,cols


def train_test(X ,y ,cols):
    X_train ,X_test ,y_train ,y_test = train_test_split(X ,y ,test_size = .25 ,random_state = 33)
    mm = MinMaxScaler().fit(X_train)

    scalerfile = 'scaler.sav'
    pickle.dump(mm ,open(scalerfile ,'wb'))

    X_train = pd.DataFrame(mm.transform(X_train))
    X_train.columns = cols
    X_test = pd.DataFrame(mm.transform(X_test))
    X_test.columns = cols

    return X_train ,X_test ,y_train ,y_test


def split_dataframe(df):
    chunk_size = df.shape[0] // 3
    list_of_dfs = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    return list_of_dfs



dataframe = preprocessing(read_data(path))
X,y,columns = oversampling(dataframe)
X_train ,X_test ,y_train ,y_test = train_test(X,y,columns)


list_of_dfs = split_dataframe(X_test)

for i, df in enumerate(list_of_dfs):
    df.to_csv(f'test_{i}.csv', index=False)

models_list = [
    "Nearest Neighbors" ,
    "XGBoost" ,
    "Logistic Regression" ,
    "Random Forest" ,
    "Neural Net" ,
    "AdaBoost" ,
    "LightGBM" ,
    "SVC RBF"
]

classifiers = [
    KNeighborsClassifier(2) ,
    xgb.XGBClassifier() ,
    LogisticRegression(fit_intercept = False ,C = 1e12 ,solver = 'liblinear') ,
    RandomForestClassifier(max_depth = 5 ,n_estimators = 200 ,max_features = 1) ,
    MLPClassifier(alpha = 0.0001 ,max_iter = 1000) ,
    AdaBoostClassifier() ,
    lgb.LGBMClassifier() ,
    SVC(gamma = 2 ,C = 1)
]



results = []
names = []
maxi = 0


for idx ,name in enumerate(models_list):
    model = classifiers[idx]
    start = time.time()
    mlflow.lightgbm.autolog()
    mlflow.xgboost.autolog()
    mlflow.sklearn.autolog()
    end = time.time()

    with mlflow.start_run(run_name = name ,nested = True):
        model_uri = "runs:/{}/{}".format(mlflow.active_run().info.run_id ,name)
        model.fit(X_train ,y_train)
        names.append(name)
        predictions = model.predict(X_test)

        mlflow.log_metric("Accuracy" ,accuracy_score(y_test, predictions))
        mlflow.log_metric("Precision" ,precision_score(y_test ,predictions))
        mlflow.log_metric("Recall" ,recall_score(y_test ,predictions))

    if accuracy_score(y_test, predictions) > maxi:
        maxi = accuracy_score(y_test, predictions)
        best_model = [name ,model]
    print("%s: Accuracy: %f --- Precison: %f --- Recall: %f (run time: %f)" % (
        name ,accuracy_score(y_test, predictions) ,precision_score(y_test ,predictions) ,recall_score(y_test ,predictions) ,end - start))

print(best_model)

pickle.dump(best_model[1] ,open('best_model.pkl' ,'wb'))

display = PrecisionRecallDisplay.from_estimator(
    best_model[1] ,X_test ,y_test ,name = best_model[0]
)
# display.ax_.set_title("Precision-Recall curve")
# plt.savefig("Precision-Recall_Curve_ChurnPrediction.svg")

cm = confusion_matrix(y_test ,predictions)
cm_matrix = pd.DataFrame(data = cm ,columns = ['Actual Positive:1' ,'Actual Negative:0'] ,
                         index = ['Predict Positive:1' ,'Predict Negative:0'])

# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues')
# plt.savefig('Confusion_Matrix_ChurnPrediction.svg')
# best_model[1].booster_.save_model('raw.json')


# for learning_rate in (0.05, 0.09):
#   for max_depth in (-5, 0, 5):
#     model = lgb.LGBMClassifier(learning_rate=learning_rate,max_depth=max_depth)
#
#     name = best_model[0]
#
#
#
#
#     with mlflow.start_run(run_name=name + '-FineTuned', nested = True):
#             model.fit(X_train ,y_train)
#             predictions = model.predict(X_test)
#             # mlflow.log_param('Learning_rate' ,learning_rate)
#             # mlflow.log_param('max_depth' ,max_depth)
#             # # mlflow.lightgbm.autolog()
#             mlflow.log_metric("Accuracy" ,maxi)
#             mlflow.log_metric("Precision" ,precision_score(y_test ,predictions))
#             mlflow.log_metric("Recall" ,recall_score(y_test ,predictions))
#             # model_uri = "runs:/{}/{}".format(mlflow.active_run().info.run_id ,name)
#             # mlflow.register_model(model_uri ,name)


# model = best_model[1]
# # model = DecisionTreeClassifier(max_depth=5)
# model = model.fit(X_train, y_train)
# predictions = model.predict(X_test)
#
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
#
# print("Accuracy score %f" % accuracy_score(y_test, predictions))
# print(classification_report(y_test, predictions))
#
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, predictions))
#
# recall_score(y_test, predictions)
#
# from sklearn.metrics import PrecisionRecallDisplay
#
# display = PrecisionRecallDisplay.from_estimator(
#     model, X_test, y_test, name="Decision Trees"
# )
# _ = display.ax_.set_title("2-class Precision-Recall curve")
#
# from sklearn.metrics import matthews_corrcoef
#
# matthews_corrcoef(y_test, predictions)
#
# import eli5
# eli5.explain_weights(model)
#
# !pip install shap
#
#


# import shap
# model = best_model[1]
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)
# explainer = shap.TreeExplainer(model, X_test)
# shap_values = explainer(X_test, check_additivity=False)
# shap.plots.beeswarm(shap_values=shap_values)
# plt.savefig('beeswaram_ChurnPrediction.svg')
#
# explainer = TreeExplainer(model)
# sv = explainer(X_train)
# exp = Explanation(sv.values[:,:,1],
#                   sv.base_values[:,1],
#                   data=X_train.values,
#                   feature_names=X_train.columns)
# idx = 0
# shap.plots.waterfall(exp[idx])
#
# shap.plots.force(exp[0], matplotlib=True)
#
#
# shap.initjs()
#
#
