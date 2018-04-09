from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical, multi_gpu_model
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
import keras.backend as K
from gensim.models import KeyedVectors
import numpy as np
import gc
from sys import getsizeof
import os

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def read_data(infile_path):
	infile_lines = open(infile_path).readlines()
	source_sents = []
	target_sents = []
	for line in infile_lines:
		[source_sent, target_sent] = line.strip().split("\t")
		source_sents.append(source_sent.strip().split())
		target_sents.append(target_sent.strip().split())
	max_length = max([len(x) for x in source_sents])
	source_sents_pad = []
	target_sents_pad = []
	for sent in source_sents:
		sent_pad = sent + ["<PAD>" for i in range(max_length - len(sent))]
		source_sents_pad.append(sent_pad)
	for sent in target_sents:
		sent_pad = ["<s>"] + sent + ["<PAD>" for i in range(max_length - len(sent))]
		target_sents_pad.append(sent_pad)
	return (source_sents_pad, target_sents_pad)

def prepare_embedding_matrix(vocab):
	emb_matrix = np.zeros((len(vocab.keys()), 200))
	for word in vocab:
		index = vocab[word]
		if word in wv_model:
			vector = wv_model[word]
			emb_matrix[index,:] = vector
		else:
			emb_matrix[index,:] = wv_model["UNK"]
	return emb_matrix

def collect_vocab(sents):
	vocab = {}
	i = 0
	for sent in sents:
		for word in sent:
			if word not in vocab:
				vocab[word] = i
				i+=1
	return vocab

def sents2ints(sents, vocab):
	int_sents = []
	for sent in sents:
		int_sent = [vocab[word] for word in sent]
		int_sents.append(int_sent)
	return int_sents

def ints21hot(int_sents, nclasses, max_length):
	sents_1hot = np.zeros((len(int_sents), max_length, nclasses), dtype = 'float32')
	for i,int_sent in enumerate(int_sents):
		for j,int_ in enumerate(int_sent):
			sents_1hot[i,j,int_] = 1.0 
	return sents_1hot

def word_ints_2_char_ints(sent_ints, vocab):
	int2word = dict((index, word) for word, index in vocab.items())
	word_max_len = max([len(word) for word, index in vocab.items()])
	new_vocab = {}
	for word in vocab:
		new_word = word + "@"*(word_max_len - len(word))
		new_vocab[new_word] = vocab[word]
	int2word_new = dict((index, word) for word, index in new_vocab.items())
	char_vocab = {}
	i = 0
	for word in new_vocab.keys():
		for char in word:
			if char not in char_vocab:
				char_vocab[char] = i
				i += 1
	char_ints = []
	for sent_int in sent_ints:
		all_sent = []
		for word_int in sent_int:
			all_word = []
			word = int2word_new[word_int]
			for char in word:
				all_word.append(char_vocab[char])
			all_sent.append(all_word)
		char_ints.append(all_sent)
	return (char_ints, word_max_len, char_vocab)
	
wv_model = KeyedVectors.load_word2vec_format('/home/sakakini/adr-detection-parent/large-files/word_vectors/PubMed-and-PMC-w2v.bin', binary=True)
infile_path = "/home/sakakini/adr-detection-parent/large-files/datasets/ADE_NER_All.txt"
latent_dim = 256
batch_size = 1000
epochs = 300
n_instances = 10000
char_emb_size = 25
window_size = 7

(source_sents, target_sents) = read_data(infile_path)
source_sents = source_sents[:n_instances]
target_sents = target_sents[:n_instances]
source_vocab = collect_vocab(source_sents)
target_vocab = collect_vocab(target_sents)
source_sents_ints = sents2ints(source_sents, source_vocab)
target_sents_ints_delayed = sents2ints(target_sents, target_vocab)

(source_char_ints, max_word_len, char_vocab) = word_ints_2_char_ints(source_sents_ints, source_vocab)
char_vocab_size = len(char_vocab.keys())

target_sents_ints = [sent[1:] for sent in target_sents_ints_delayed]
target_sents_ints_delayed = [sent[:-1] for sent in target_sents_ints_delayed]

source_vocab_size = len(source_vocab.keys())
target_vocab_size = len(target_vocab.keys())

emb_matrix = prepare_embedding_matrix(source_vocab)

max_length = max([len(x) for x in target_sents_ints])

target_sents_1hot = ints21hot(target_sents_ints, target_vocab_size, max_length)

"""
model_input_source_chars = Input(shape = (max_length, max_word_len), dtype='int32')
embedding_layer_char = Embedding(char_vocab_size, char_emb_size)
conv_layer = Conv2D(char_emb_size, window_size, padding='same', activation = 'tanh', data_format = 'channels_last')
pooling_layer = MaxPooling2D(pool_size=(1,max_word_len), data_format = 'channels_last')
reshape_layer = Reshape((max_length,char_emb_size))

single_char_emb = embedding_layer_char(model_input_source_chars)
conv_output = conv_layer(single_char_emb)
char_emb = pooling_layer(conv_output)
char_emb = reshape_layer(char_emb)
"""

"""
print(embedding_layer_char.input_shape)
print(embedding_layer_char.output_shape)
"""

model_input_source = Input(shape = (max_length,), dtype='int32')
model_input_target = Input(shape = (max_length,), dtype='int32')
embedding_layer_fixed = Embedding(source_vocab_size, 200, weights = [emb_matrix], trainable=False)
embedding_layer_trainable = Embedding(input_dim = source_vocab_size, output_dim = 300)
embedding_layer_target = Embedding(output_dim=30, input_dim=target_vocab_size)
source_embedding_fixed = embedding_layer_fixed(model_input_source)
source_embedding_trainable = embedding_layer_trainable(model_input_source)
target_embedding = embedding_layer_target(model_input_target)
LSTM_layer= Bidirectional(LSTM(latent_dim, return_sequences = True))
#lstm_input = concatenate([source_embedding_fixed, source_embedding_trainable, char_emb])
lstm_input = concatenate([source_embedding_fixed, source_embedding_trainable])
dropout_layer_1 = Dropout(0.9)
lstm_input = dropout_layer_1(lstm_input)
lstm_output = LSTM_layer(lstm_input)
dense_layer = Dense(target_vocab_size, activation='softmax')
dense_input = concatenate([lstm_output, target_embedding])
dropout_layer_2 = Dropout(0.9)
dense_input = dropout_layer_2(dense_input)
prediction = dense_layer(dense_input)
#model = Model(inputs = [model_input_source, model_input_target, model_input_source_chars], outputs = [prediction])
model = Model(inputs = [model_input_source, model_input_target], outputs = [prediction])
optimizer = RMSprop(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
#model.fit([np.array(source_sents_ints), np.array(target_sents_ints_delayed), np.array(source_char_ints)], [np.array(target_sents_1hot)], batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.fit([np.array(source_sents_ints), np.array(target_sents_ints_delayed)], [np.array(target_sents_1hot)], batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save("model.h5")

# Define Bidirectional model

model = Model(inputs = [model_input_source], outputs = [lstm_output])
model.save("bidirectional.h5")

# Define Prediction model

lstm_state = Input(shape = (1,latent_dim*2,), dtype="float32")
previous_tag = Input(shape = (1,), dtype="int32")
previous_tag_embedding = embedding_layer_target(previous_tag)
decoder_dense_input = concatenate([lstm_state, previous_tag_embedding])
decoder_prediction = dense_layer(decoder_dense_input)
model = Model(inputs = [lstm_state, previous_tag], outputs = [decoder_prediction])
model.save("prediction_model.h5") 

