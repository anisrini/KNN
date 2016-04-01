import pandas as pd
import helpers
import cdist 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def knn(trains, test, kvals, loo_flag):
	print "Classification"

	for k in range(len(kvals)):
		print "K: \n", kvals[k]
		knn = cdist.KNNClassifier(kvals[k])
		for i in range(len(trains)):
			for j in range(len(trains[i])):
				print "2-Fold"
				print "i, j ", i, j
				print
				skf = StratifiedKFold(trains[i][j]['labels'], n_folds=2)
				for train_index, test_index in skf:
					fold_train = trains[i][j].ix[train_index]
					fold_test = trains[i][j].ix[test_index]

					fold_train.index = range(len(fold_train))
					fold_test.index = range(len(fold_test))
					
					knn.train(fold_train.iloc[:,:-1], fold_train['labels'])

					print "Accuracy: ", accuracy_score(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
					print "Confusion Matrix: \n", confusion_matrix(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
					print

				print "5-Fold"
				print "i, j ", i, j
				print
				skf = StratifiedKFold(trains[i][j]['labels'], n_folds=5)
				for train_index, test_index in skf:
					fold_train = trains[i][j].ix[train_index]
					fold_test = trains[i][j].ix[test_index]

					fold_train.index = range(len(fold_train))
					fold_test.index = range(len(fold_test))
					
					knn.train(fold_train.iloc[:,:-1], fold_train['labels'])

					print "Accuracy: ", accuracy_score(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
					print "Confusion Matrix: \n", confusion_matrix(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
					print

				if loo_flag == 1:
					print "Leave One Out: "
					print "i, j ", i, j
					print
					loo = LeaveOneOut(len(trains[i][j].iloc[:,:-1]))
					for train_index, test_index in loo:
						fold_train = trains[i][j].ix[train_index]
						fold_test = trains[i][j].ix[test_index]

						fold_train.index = range(len(fold_train))
						fold_test.index = range(len(fold_test))
						
						knn.train(fold_train.iloc[:,:-1], fold_train['labels'])

						print "Accuracy: ", accuracy_score(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
						print "Confusion Matrix: \n", confusion_matrix(fold_test['labels'], knn.test(fold_test.iloc[:,:-1]))
						print

				print
