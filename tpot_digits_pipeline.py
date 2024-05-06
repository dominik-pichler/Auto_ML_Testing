import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score


# kick:
kick_dimRed_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/kick/clean/kick_train_preprocessed.csv")
kick_dimRed_15_train_y = kick_dimRed_15_train["IsBadBuy"]
kick_dimRed_15_train_x = kick_dimRed_15_train.drop("IsBadBuy", axis=1)

kick_dimRed_15_test = pd.read_csv(  "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/kick/clean/kick_test_preprocessed.csv")
kick_dimRed_15_test_y = kick_dimRed_15_test["IsBadBuy"]
kick_dimRed_15_test_x = kick_dimRed_15_test.drop("IsBadBuy", axis=1)

# Average CV score on the training set was: 0.904112055507327
exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="constant", loss="perceptron", penalty="elasticnet", power_t=0.0)),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=8, min_samples_split=4, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(kick_dimRed_15_train_x, kick_dimRed_15_train_y)
results = exported_pipeline.predict(kick_dimRed_15_test_x)

print("F1 " + str(f1_score(y_true=kick_dimRed_15_test_y, y_pred=results, average="binary", pos_label=1)))
print("Accuracy " + str(accuracy_score(y_true=kick_dimRed_15_test_y, y_pred=results)))
print("Precision " + str(precision_score(y_true=kick_dimRed_15_test_y, y_pred=results, pos_label=1)))
print("Recall " + str(recall_score(y_true=kick_dimRed_15_test_y, y_pred=results, pos_label=1)))
