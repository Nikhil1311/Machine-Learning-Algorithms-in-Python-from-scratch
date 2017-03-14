#############################################################################################################################
###############################                     SIMPLE LINEAR REGRESSION               ##################################
#############################################################################################################################

#Simple Linear Regression is applied to Swedish Insurance Dataset
#x values are total number of claims and y values are total payment of claims in thousands

from math import sqrt
import csv
import random

#to calculate mean
def mean(points):
	return sum(points)/float(len(points))

#to calculate variance
def variance(points, mean):
	return sum([(x-mean)**2 for x in points])

#to calculate covariance
def covariance(x, x_mean, y, y_mean):
	covar = 0
	for i in range(len(x)):
		covar += (x[i]-x_mean) * (y[i]-y_mean)
	return covar

#to calculate weights
def weights(dataset):
	x = [row[0] for row in dataset]
	y = [column[1] for column in dataset]
	x_mean = mean(x)
	y_mean = mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

#Simple linear regression model --> Y = B0 + B1*X 
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = weights(train)
	for each_row in test:
		ycap = b0 + b1 * each_row
		predictions.append(ycap)
	return predictions

#Root mean squared error metric
def root_mean_squared(actual, predicted):
	error = 0
	for i in range(len(actual)):
		pred_error = actual[i] - predicted[i]
		error += (pred_error**2)
	mean_error = error / float(len(actual))
	return sqrt(mean_error)

#to evaluate simple_linear_regression on training dataset
def evaluate(dataset, algo, split):
	train, test = train_test_split(dataset, split)
	test_list = [row[0] for row in test]
	predicted = algo(dataset, test_list)
	actual = [row[1] for row in test]
	rmse = root_mean_squared(actual, predicted)
	return rmse

#to load csv file
def load(filename):
	dataset = list()
	with open(filename, 'r') as file:
		read = csv.reader(file)
		for row in read:
			if not row:
				continue
			dataset.append(row)
	return dataset

#to split dataset into train and test sets
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = random.randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy	

random.seed(1)
# load and prepare data
filename = 'insurance.csv'
dataset = load(filename)

#converting string to float
for i in range(len(dataset)):
	for j in range(len(dataset[i])):
		dataset[i][j] = float(dataset[i][j])

# evaluate algorithm
split = 0.6
rmse = evaluate(dataset, simple_linear_regression, split)
print'RMSE: %.3f' % (rmse)	
