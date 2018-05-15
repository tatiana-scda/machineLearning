import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#------------------------------------------------------
#hyperparameters settings

#1. Gradient Descent: batch = 5000
#2. Stochastic Gradient Descent: batch = 1
#3. Mini-Batch: batch = 10 ~ 50
batch_size = 50
neurons_in_hl = 100 #25, 50 or 100
learning_rate = 0.05 #Varies between 0.05, 1 and 10

#------------------------------------------------------

nmist_df = pd.read_csv('mnist_dataset.csv', header=None)

#parsing the dataset
labels = nmist_df.iloc[:, :1] #here i'm setting all the lines and then column from initial to one (so, only the comun zero)
features = nmist_df.iloc[:, 1:] #setting now all the rest of mnist

#creating one-hot
mnist_classes = np.eye(10, dtype=int)[[item for sublist in labels.values for item in sublist]]
mnist_features = features.values

#input and output
x = tf.placeholder(tf.float32, [None, 784], name='ph_features')
y = tf.placeholder(tf.float32, [None, 10], name='ph_output')

#weight and bias from the hidden layer
w = tf.Variable(tf.random_normal([784, neurons_in_hl], mean=0, stddev=1 / np.sqrt(784)), name='weights')
b = tf.Variable(tf.random_normal([neurons_in_hl], mean=0, stddev=1 / np.sqrt(784)), name='biases')

#weight and bias from the outup layer
wo = tf.Variable(tf.random_normal([neurons_in_hl, 10], mean=0, stddev=1/np.sqrt(784)), name='weightsOut')
bo = tf.Variable(tf.random_normal([10], mean=0, stddev=1/np.sqrt(784)), name='biasesOut')

#hidden layer ajust function using sigmoid and matmul for matrix multiplication
hl = tf.nn.sigmoid((tf.matmul(x, w) + b),name='activationLayer')

#compute the output layer with updated values for weight ans bias
out = tf.matmul(hl, wo) + bo

#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = out))

#corrections
correct_predictions = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) 

#optmizer
update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#initialization of all variables
initial = tf.global_variables_initializer()

print("initial setup: batch: %d \nneurons in the hidden layer: %d \nlearning rate: %.2f" %(batch_size, neurons_in_hl, learning_rate))

#launch a session to run 
with tf.Session() as sess:
	sess.run(initial) #initializes variables

	plot_data = []

	for epoch in range(100):

		#train with each example
		total_batch = int(5000/batch_size)
		for i in range (total_batch):

			start = i*batch_size #separating each batch considering the total and how many times we will do this process
			end = (i+1)*batch_size

			update.run(feed_dict = {x: mnist_features[start:end], y: mnist_classes[start:end]})
		acc = accuracy.eval(feed_dict = {x: mnist_features, y: mnist_classes})
		print("epoch: " + str(epoch) + ", train accuracy: " + str(acc))
		plot_data.append(acc)
	plt.plot(plot_data)	
	plt.show()
	#plt.plot(x, y);
	#plt.plot(x, acc);

sess.close()