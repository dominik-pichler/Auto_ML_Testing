from tpot import TPOTClassifier
import pandas as pd



# diabetes


# kick:
kick_dimRed_15_train = pd.read_csv("/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/kick/clean/kick_train_preprocessed.csv")
kick_dimRed_15_train_y = kick_dimRed_15_train["IsBadBuy"]
kick_dimRed_15_train_x = kick_dimRed_15_train.drop("IsBadBuy", axis=1)

kick_dimRed_15_test = pd.read_csv(  "/Users/dominikpichler/Documents/Git/ml-assignments/ml-assignments/data/kick/clean/kick_test_preprocessed.csv")
kick_dimRed_15_test_y = kick_dimRed_15_test["IsBadBuy"]
kick_dimRed_15_test_x = kick_dimRed_15_test.drop("IsBadBuy", axis=1)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(kick_dimRed_15_train_x, kick_dimRed_15_train_y)
print(tpot.score(kick_dimRed_15_test_x, kick_dimRed_15_test_y))
tpot.export('tpot_digits_pipeline.py')
