import pandas as pd
import numpy as np
import scipy as sp

class KNNClassifier:
	def __init__(self, k):
		self.df = pd.DataFrame()
		self.labels = pd.DataFrame()
		self.k = k

	def train(self, df, labels):
		self.df = df
		self.labels = labels

	def classify(self, x):
		x = pd.DataFrame(x)
		x.columns = ['val']
		x = x.sort(['val'], axis = 0)
		x = x.iloc[:self.k]
		res = pd.DataFrame(self.labels.ix[x.index.values])
		res.columns = ['val']
		return sp.stats.mstats.mode(res['val'])[0][0]

	def test(self, test):
		dist = sp.spatial.distance.cdist(self.df, test, 'euclidean')
		dist = pd.DataFrame(dist)
		return dist.apply(self.classify, axis = 0)