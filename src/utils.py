import os
import tarfile
import cPickle as pickle
import numpy as np
from subprocess import call

def load_batches(filename):
	#Load CIFAR-10 batches where data is shape (10000,32,32,3)
	with open(filename,'rb') as f:
		dict = pickle.load(f)
		X=dict['data'].reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
		Y=np.array(dict['labels'])
		return X,Y

def load_data():
	#check if data is downloaded
	def check_data():
		new_path = os.path.join(
				os.path.split(__file__)[0],
				"..",
				"data",
				"cifar-10-batches-py")
		if (not os.path.exists(new_path)):
			print 'Download data using data/get_datasets.sh'

	check_data()
	#load training data
	x = []
	y = []
	for i in range(1,6):
		filename = os.path.join('data/cifar-10-batches-py','data_batch_%d' %(i))
		X,Y = load_batches(filename)
		x.append(X)
		y.append(Y)
	X_tr = np.concatenate(x)
	Y_tr = np.concatenate(y)
	del X,Y

	#load test data
 	X_te, Y_te = load_batches(os.path.join('data/cifar-10-batches-py', 'test_batch'))		

	return X_tr,Y_tr,X_te,Y_te
	
def preproc_data(x):
	#reshape data
	x = np.reshape(x,(x.shape[0],-1))
	#subtract mean
	x -= np.mean(x,axis= 0)
	#add 1 at the end for bias
	x = np.hstack([x,np.ones((x.shape[0],1))])
	return x

