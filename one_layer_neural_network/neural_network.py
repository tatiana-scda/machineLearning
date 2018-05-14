import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------
#parameters settings

#1. Gradient Descent: batch = 5000
#2. Stochastic Gradient Descent: batch = 1
#3. Mini-Batch: batch = 10 ~ 50
batch_size = 5000
neurons_in_hl = 10 #25, 50 or 100
learning_rate = 0.01 #Varies between 0.05, 1 and 10

#------------------------------------------------------

nmist_df = pd.read_csv('nmist_dataset.csv', header=None)

#setting data
labels = nmist_df.iloc[:, :1] #here i'm setting all the lines and then column from initial to one (so, only the comun zero)
features = nmist_df.iloc[:, 1:] #setting now all the rest of mnist

#input and output
x = tf.placeholder(tf.float32, [None, 784], name='ph_features')
y = tf.placeholder(tf.float32, [None, 10], name='ph_output')

#weight and bias from the hidden layer
w = tf.Variable(tf.truncated_normal([784, neurons_in_hl], mean=0, stddev=1 / np.sqrt(784)), name='weights')
b = tf.Variable(tf.truncated_normal([neurons_in_hl], mean=0, stddev=1 / np.sqrt(784)), name='biases')

#weight and bias from the outup layer
wo = tf.Variable(tf.random_normal([neurons_in_hl, 10], mean=0, stddev=1/np.sqrt(784)), name='weightsOut')
bo = tf.Variable(tf.random_normal([10], mean=0, stddev=1/np.sqrt(784)), name='biasesOut')

#hidden layer ajust function
hl = tf.nn.sigmoid((tf.matmul(y,wo)+bo),name='activationLayer')

#compute the output layer
out = tf.matmul(hl,wo) + bo

#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = out))

#corrections
correct_predictions = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) 

#optmizer
update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# initialization of all variables
initial = tf.global_variables_initializer()

print("initial setup: batch: %d \nneurons in the hidden layer: %d \nlearning rate: %d" %(batch_size, neurons_in_hl, learning_rate))

# Launch a session to run 
with tf.Session() as sess:
	sess.run(initial) # Initializes variables

	for epoch in range(100):
		#Train with each example
		for i in range(len(mnist[0])):
			start = i
			end = i + batch_size
			x = features[start:end]
			y = labels[start:end]
			update.run(feed_dict = {x: batch[0], y: batch[1]})
		accuracy = accuracy.eval(feed_dict = {x: mnist[0], y: mnist[1]}) #0 feat 1 label
		print("epoch: " + epoch + ", train accuracy = " + accuracy)
	plt.plot(x, y);
	plt.plot(x, accuracy);

sess.close()