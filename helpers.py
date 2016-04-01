import random
import pandas as pd

def sample(df):
	size = len(df)
	rows = random.sample(range(size), int(0.2 * size))

	test = df.ix[rows]
	train = df.drop(rows)

	return train, test

def subset(train):
	d = [20, 50, 80, 100]
	size = len(train)
	trains = [[], [], [], [train]]

	for i in range(len(trains)):
		for j in range(5):
			rows = random.sample(range(size), int((d[i]/100.0) * size))
			temp_train = train.ix[rows]
			temp_train.index = range(len(temp_train))
			trains[i].append(temp_train)

	return trains

#Function to normalize data
def normalize(df):
	return (df - df.mean())/ df.std()


def normalize_trains(trains):
	for i in range(len(trains)):
		for j in range(len(trains[i])):
			temp_train_labels = trains[i][j]['labels']
			temp_train = trains[i][j].iloc[:,:-1]
			temp_train = normalize(temp_train)
			temp_train["labels"] = temp_train_labels
			trains[i][j] = temp_train

	return trains