###############################################################
# Filename: RNN_project.py
# Version: 1.0
# Author: Trent Rogers
# Email: trentr314@gmail.com
# Original completion date: 2020-11-25
# Last updated: 2022-08-27
# Description:
# 	This program takes over 5200 days of closing prices of IBM
# 	and runs them through a simple Jordan recurrent neural
# 	network.
###############################################################


# needed for the matrices involved in the neural net
import numpy as np
# used to read data initially
import pandas as pd
# for graphing results
import matplotlib.pyplot as plt
# data is pulled from my GitHub
import urllib.request
# import datetime so that we can record the time the program starts and finishes
import datetime
# used to check for overflow
import math
# used to exit if overflow is encountered
import sys
# to log results of each run
import csv


# record the time the program starts running
# the start_time variable is later used to calculate running time
start_time = datetime.datetime.now()


'''  ************************* hyperparameters *************************  '''

# the number of nodes in the hidden layer
hidden_size = 6
# the learning rate
lr = 0.00000000003
# the number of passes to be made over the whole dataset
n_epochs = 200
# the number of days of data that are used in a round of forward predictions
# before a round of backpropogation
batch_size=10
# out of 5298 days of data
# a bit of the earliest data is cropped off so that it's neatly divisible into batches
epoch_size=5200
# the number of batches (number of backpropogations per epoch)
# needs to be an integer, but is calculated by division, so you need to be sure that
# epoch_size is a multiple of batch_size
n_batches = int(epoch_size/batch_size)


# file is a csv created from data pulled from alphavantage, and hosted on my GitHub publicly.
openurl = urllib.request.urlopen("https://raw.githubusercontent.com/trentr314/RNN_Project/master/data/IBM_full_daily_adjusted.csv")

df = pd.read_csv(openurl)
# we use only the closing price
close_prices = df['close']
close_prices.to_numpy()
# crops the first few data points to get the wanted epoch size
close_prices = close_prices[-epoch_size:]
# necessary to shape the data properly
close_prices = close_prices.to_numpy()[:, np.newaxis]
# shape the matrix so that we can iterate through batches
close_prices = close_prices.reshape((n_batches,batch_size,1))
# a copy must be made to prevent the input and output from pointing to the same thing
myinput = close_prices.copy()
# the output will be changed on forward passes; this is just a convenient initialization
out = close_prices.copy()


# random initialization of the weights that transform input to values at the hidden layer
w_in = np.random.rand(1,hidden_size)
# random initialization of the weights that transform values at the hidden layer to output
w_out = np.random.rand(hidden_size, 1)
# random initialization of the weights that transform output to recurrent input
w_rec = np.random.rand(1,hidden_size)
# the matrix that holds the values of the hidden nodes, which will need to record the hidden node values
# for each iteration of a batch in order to perform backpropogation
inner_nodes = np.random.rand(batch_size,hidden_size)
# the biases for the hidden layer
inner_biases = np.random.rand(1,hidden_size)
# the bias for the output layer
out_bias = np.random.rand(1,1)


# creates forward predictions for a given batch
def forward(out, batch_num, batch_size):

	# iterates through a batch
	for i in range(batch_size - 1):

		# for the first iteration of a batch, out contains actual data instead of predicted data
		# this is an impurity of this implementation, but its significance decreases as batch size increases.
		inner_nodes[i,...] = myinput[batch_num,i,...] @ w_in + out[batch_num,i,...] @ w_rec + inner_biases

		# creates a prediction that will be used as recurrent input to the next iteration in the batch
		out[batch_num,i+1,...] = inner_nodes[i,...] @ w_out + out_bias

	return(out)


# uses calculated gradients to backpropogate error and change the weights
# takes quite a few arguments to avoid globals
# really the heart of the algorithm
def backward(w_out,lr,inner_nodes,out,myinput,w_in,w_rec,inner_biases,out_bias,batch_num):
	w_out -= lr * ( ( inner_nodes.T @ (out[batch_num,...] - myinput[batch_num,...] ) ) )
	w_in -= lr * ( ( myinput[batch_num,...].T) @ ( out[batch_num,...] - myinput[batch_num,...] ) @ (w_out.T) )
	w_rec -= lr * ( ( myinput[batch_num,...].T) @ (out[batch_num,...] - myinput[batch_num,...]) @ (w_out.T) )
	inner_biases -= lr * ( sum( out[batch_num,...]-myinput[batch_num,...] ) )
	out_bias -= lr * ( sum( out[batch_num,...] - myinput[batch_num,...] ) )

def mse(out,myinput):
	# flattening puts all the data back in one row instead of dividing by batches
	out_row = out.flatten()
	input_row = myinput.flatten()
	# elementwise subtraction
	diff = np.subtract(out_row, input_row)
	# elementwise squaring
	diff2 = np.square(diff)
	# adds all the squared differences in the array
	sum_diff2 = sum(diff2)
	# returns the mean squared error
	return(sum_diff2/epoch_size)


# starts the training
# will run through the whole dataset n_epochs times
for epoch in range(n_epochs):

	# runs a forward prediction and a backpropagation for every batch
	for batch_num in range(n_batches):

		# creates predictions
		out = forward(out, batch_num, batch_size)

		# runs backprop
		backward(w_out,lr,inner_nodes,out,myinput,w_in,w_rec,inner_biases,out_bias,batch_num)

	# calculates the MSE for the epoch
	epoch_mse = mse(out,myinput)

	# sometimes overflow is encountered with bad hyperparameters, or even good ones
	if(math.isnan(epoch_mse)):
		print("Overflow occurred.  Please try again.")
		sys.exit()

	# shown on the graph to demonstrate the model's improvement over epochs
	if(epoch==100):
		out_row = out.reshape((1,5200))
		plt.plot(out_row[0,-50:], 'b-^', linewidth=0.25)


	# This gives stdout regular approximations of how much longer you have to wait
	# The first prediction is particularly inaccurate but improves quickly
	if(epoch%24==0):
		epoch_time = datetime.datetime.now() - start_time
		projected_time = (epoch_time * n_epochs / (epoch+1)) - epoch_time
		projected_time -= datetime.timedelta(microseconds=projected_time.microseconds)
		print('projected remaining runtime: ' + str(projected_time))
		# flush is because otherwise nothing will print until PyPlot is done with its graph
		sys.stdout.flush()


# logs the time the program ends and the time it took to run
end_time = datetime.datetime.now()
run_time = end_time - start_time
run_time -= datetime.timedelta(microseconds=run_time.microseconds)

# shapes the data so that it's easier to plot (all in one row instead of divided into batches)
out_row = out.reshape((1,5200))
myinput_row = myinput.reshape((1,5200))

# logs the features of the run in a .csv file
with open('RNN_project_log.csv', 'a', newline='') as log:
	log_info = [lr, n_epochs, hidden_size, batch_size, run_time, epoch_mse]
	logwriter = csv.writer(log)
	logwriter.writerow(log_info)

# formats and displays the data about the run, showing its accuracy on the last 50 days of the last epoch
# also shows all the hyperparameters used for convenience and record-keeping
plt.plot(out_row[0,-50:], 'r-s', linewidth=0.5)
plt.plot(myinput_row[0,-50:], 'g-o', linewidth=0.5)
plt.ylabel('IBM closing price')
plt.title('blue triangles = 100th epoch predictions\nred squares = final predictions\ngreen circles = actual')
plt.xlabel('past 50 days')
plt.annotate('hyperparameters:\n' \
	+ 'learning rate: ' + str(lr) \
	+ '\nhidden size: ' + str(hidden_size) \
	+ '\nbatch size: ' + str(batch_size) \
	+ '\nn_epochs: ' + str(n_epochs) \
	+ '\nrunning time: ' + str(run_time) \
	+ '\nMSE: ' + str(round(epoch_mse, 2)), \
	xy=(0.85,0.85), xycoords='axes fraction')
plt.show()
