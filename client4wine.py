import pandas as pd
import helpers
import knn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut

kvals = (3, 5, 7, 11, 13, 17)
loo_flag = 1

df = pd.read_csv("data/wine.data", header = None)

train, test = helpers.sample(df)

train.index = range(len(train))
test.index = range(len(test))

train_labels = train.iloc[:,0]
train = train.iloc[:,1:]
train['labels'] = train_labels

trains = helpers.subset(train)

test_labels = test.iloc[:,0]
test = helpers.normalize(test.iloc[:,1:])
test['labels'] = test_labels

trains = helpers.normalize_trains(trains)

knn.knn(trains, test, kvals, loo_flag)


