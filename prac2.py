# prac2.py

from ted_data import ted_talks_and_labels 
import tensorflow as tf
import argparse 
import urllib
import zipfile
import os
from nltk.corpus import stopwords
from collections import Counter 
import numpy as np
import random


def get_glove_embeddings_min(embedding_size,vocab): #returns an embedding dict
	if not os.path.isfile('glove.6B.zip'):
		urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
	
	z = zipfile.ZipFile('glove.6B.zip', 'r')
	lines = z.open('glove.6B.'+str(embedding_size)+'d.txt', 'r')
	count = 0
	glove_embedding = {'UNK': np.zeros(int(embedding_size))}
	for line in lines:
		word_vec = line.decode("utf-8").strip().split(' ')
		word  = word_vec[0]
		if word in vocab:
			vector = [float(i) for i in word_vec[1:]]
			glove_embedding[word] = vector
	return glove_embedding


def reduce_talks(talks_text, vocab_size): #returns list of redcued talks
	sstopwords = set(stopwords.words("english"))
	without_stop_words = [[word for word in talk if word not in sstopwords] for talk in talks_text]
	counter = Counter()
	for talk in without_stop_words:
		for word in talk:
			counter[word] += 1
	allowed_words = set([word for word, count in counter.most_common(vocab_size-1)])
	reduced_talks = [[word for word in talk if word in allowed_words] for talk in without_stop_words]
	return reduced_talks, allowed_words


# Returns a matrix of the vectors with a dict to represent which index a word is at in the matrix
def generate_embedding_matrix(embedding_dict):
	index_dict = {}
	embedding_matrix = []
	for i, word in enumerate(embedding_dict):
		index_dict[word] = i
		embedding_matrix.append(embedding_dict[word])
	return np.asarray(embedding_matrix), index_dict


def tokenise(talks, token_dict):
	return [[token_dict.get(word, token_dict.get('UNK')) for word in talk] for talk in talks]


def normalise(talks, length):
	inital = np.zeros([len(talks), length])
	for i, talk in enumerate(talks): 
		for j in range(length):
			inital[i][j] = talk[j%len(talk)]
	return inital	


def vectorise(talks, vocab_size):
	one_hot_talks = np.zero([len(talks),len(talks)[0], vocab_size])
	for i, talk in enumerate(talks):
		for j, index in enumerate(talk):
			if index != -1:
				one_hot_talks[i][index] = 1
	return one_hot_talks

def training_generator(paired_data, batch_size):
	while True:
		random.shuffle(paired_data)
		for i in range(len(paired_data)//batch_size):
			batch = paired_data[batch_size*i:batch_size*i+batch_size]
			yield [[item[0] for item in batch], [item[1] for item in batch]]


def main():
	parser = argparse.ArgumentParser(description='Practical 2 Classifer')
	parser.add_argument('embedding_size',help='The size of our word embeddings (50/100/200/300)')
	parser.add_argument('vocab_size',help='The size of the vocabulary')
	parser.add_argument('glove_embeddings', help='Should we use glove embeddings?')
	parser.add_argument('learnable_embeddings', help='are the embeddings learnable? 1 for yes')
	parser.add_argument('hidden_layer_size', help='The size of the hidden layer')
	parser.add_argument('dropout', help='The prob of dropout for our hidden layer')
	parser.add_argument('epochs', help='The number of epochs we will perform')
	args = parser.parse_args()

	print("\n Training with hyperparams: ")
	print("\n    Embedding size: " + str(args.embedding_size))
	print("    Vocab size: " + str(args.vocab_size))
	print("    Glove embeddings? : " + str(args.glove_embeddings))
	print("    Hidden size: " + str(args.hidden_layer_size))
	print("    Learnable embeddings: " + str(args.learnable_embeddings))
	print("    Dropout: " + str(args.dropout))
	print("    #Epochs: " + str(args.epochs) + '\n')

	# Get the data 
	talks_text, talks_labels = ted_talks_and_labels()

	# Split the data
	training = [talks_text[0:1585], talks_labels[0:1585]]	
	validation = [talks_text[1585:1585+250], talks_labels[1585:1585+250]]
	test = [talks_text[1585+250:1585+500], talks_labels[1585+250:1585+500]]

	# Reduce the size of the vocab used and remove stop words 
	reduced_texts, vocab = reduce_talks(talks_text, int(args.vocab_size))
	redcued_training = [reduced_texts, training[1]]

	# Generate the smallest embedding using the training data
	if int(args.glove_embeddings):
		glove_embedding_dict = get_glove_embeddings_min(args.embedding_size, vocab)
	else:
		glove_embedding_dict = get_glove_embeddings_min(50, vocab)

	glove_embedding_matrix, glove_index_dict = generate_embedding_matrix(glove_embedding_dict)
	# tokenise and normalise each set of data
	max_talk_length = max(map(len,training[0]))
	tokenised_training = [normalise(tokenise(redcued_training[0], glove_index_dict), max_talk_length), training[1]]
	tokenised_validation = [normalise(tokenise(validation[0], glove_index_dict), max_talk_length), validation[1]]
	tokenised_test = [normalise(tokenise(test[0], glove_index_dict), max_talk_length), test[1]]

	# Define the network

	# Data placeholders
	x = tf.placeholder(tf.int32, [None, max_talk_length])
	y = tf.placeholder(tf.int32, [None,8])

	# Embeddings 
	if int(args.glove_embeddings):
		E = tf.Variable(tf.cast(glove_embedding_matrix, tf.float32), 
			trainable=int(args.learnable_embeddings))
	else:
		E = tf.Variable(tf.truncated_normal([int(args.vocab_size), int(args.embedding_size)], stddev=1),
			trainable=int(args.learnable_embeddings))

	# Bag of means the input using the embeddigns
	X = tf.nn.embedding_lookup(E,x)
	X_sums = tf.reduce_sum(X, axis=1) # (batch_size, embedding_size)

	#X_zeros = tf.reduce_sum(x, axis=0)

	non_zero_counts = tf.reshape(tf.count_nonzero(x, axis=1),(50,1))

	doc_emeddings = tf.div(X_sums,tf.cast(non_zero_counts, tf.float32))
	#doc_emeddings = tf.reduce_mean(X, axis=1)

	W = tf.Variable(tf.random_normal([int(args.embedding_size),int(args.hidden_layer_size)], stddev=0.1))
	b = tf.Variable(tf.random_normal([int(args.hidden_layer_size)], stddev=0.1))
	h = tf.nn.dropout(tf.nn.relu(tf.matmul(doc_emeddings,W) + b),float(args.dropout))

	V = tf.Variable(tf.random_normal([int(args.hidden_layer_size),8], stddev=0.1))
	c = tf.Variable(tf.random_normal([8], stddev=0.1))
	u = tf.matmul(h,V) + c

	prediction = tf.argmax(u, dimension=1)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(u, y))
	train_step = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(cost)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	training_data = training_generator(list(zip(tokenised_training[0],tokenised_training[1])),50)
	validation_data = training_generator(list(zip(tokenised_validation[0],tokenised_validation[1])),50)
	test_data = training_generator(list(zip(tokenised_test[0],tokenised_test[1])),50)

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(u,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	validaton_acc = []
	test_acc = []
	import signal
	import time
	import matplotlib.pyplot as plt
	def sigint_handler(signum, frame):
		plt.plot(test_acc)
		plt.plot(validaton_acc)
		plt.show()
 
	signal.signal(signal.SIGINT, sigint_handler)

	print(" Starting Training: \n")
	for i in range(int(args.epochs)):
		avg = 0
		costtt = 0
		for j in range(1585//50):
			batch = next(training_data)
			_, acc, costt = sess.run([train_step,accuracy, cost], feed_dict={x: batch[0], y: batch[1]})
			avg += acc/(1585//50)
			costtt += costt/(1585//50)
		test_acc.append(avg)
		acc_valid = 0
		for j in range(5):
			batch = next(validation_data)
			acc = (sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]}))
			acc_valid += acc/5
		validaton_acc.append(acc_valid)
		#if not i % 10: 
		print("Epoch " + str(i) + "-  Train acc: " + str(avg) + " cost: " + str(costt) + " Validation acc: " +str(acc_valid))
	
	avg = 0
	batch = next(test_data)
	for i in range(5):
		acc = (sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]}))
		avg += acc/10
	print("\n \n Final test acc: " + str(avg))

	print(np.shape(V.eval(session=sess)))
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2, random_state=0)
	dataa = tsne.fit_transform(np.transpose(V.eval(session=sess)))
	print(dataa)

	vis_x = dataa[:, 0]
	vis_y = dataa[:, 1]
	labels = ["ooo","Too","oEo","TEo","ooD","ToD","oED","TED"]
	fig, ax = plt.subplots()
	ax.scatter(vis_x,vis_y)
	for i, txt in enumerate(labels):
	    ax.annotate(txt, (vis_x[i],vis_y[i]))
	plt.show()

	#plt.plot(test_acc)
	#plt.plot(validaton_acc)
	#plt.show()
main()