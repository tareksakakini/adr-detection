from keras.models import Model
from keras.layers import *
from keras.utils import to_categorical, multi_gpu_model
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
from gensim.models import KeyedVectors
import numpy as np
import gc
from sys import getsizeof
import os
from tensorflow.python.client import device_lib

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
	print("number of gpus available:",len(gpu_names))
	return len(gpu_names)


class MyCbk_full(Callback):

	def __init__(self, model):
		self.model_to_save = model

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save("full.h5")

class MyCbk_bi_ner(Callback):

	def __init__(self, model):
		self.model_to_save = model

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save("bi_ner.h5")

class MyCbk_bi_pos(Callback):

	def __init__(self, model):
		self.model_to_save = model

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save("bi_pos.h5")

class MyCbk_dense_pos(Callback):

	def __init__(self, model):
		self.model_to_save = model

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save("dense_pos.h5")

class MyCbk_dense_ner(Callback):

	def __init__(self, model):
		self.model_to_save = model

	def on_epoch_end(self, epoch, logs=None):
		self.model_to_save.save("dense_ner.h5")

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

def matricize_data(source_sents, target_sents, source_vocab, target_vocab, target_vocab_specific):
	source_sents_ints = sents2ints(source_sents, source_vocab)
	target_sents_ints_delayed = sents2ints(target_sents, target_vocab_specific)
	target_sents_ints = [sent[1:] for sent in target_sents_ints_delayed]
	target_sents_ints_delayed = [sent[:-1] for sent in target_sents_ints_delayed]
	max_length = max([len(x) for x in target_sents_ints])
	target_vocab_size_specific = len(target_vocab_specific.keys())
	target_sents_1hot = ints21hot(target_sents_ints, target_vocab_size_specific, max_length)
	return (max_length, source_sents_ints, target_sents_ints_delayed, target_sents_1hot)

def upsample(source_sents, target_sents, rate):
	new_source_sents = []
	new_target_sents = []
	for sent in source_sents:
		for i in range(rate):
			new_source_sents.append(sent)
	for sent in target_sents:
		for i in range(rate):
			new_target_sents.append(sent)
	return (new_source_sents, new_target_sents)
	
wv_model = KeyedVectors.load_word2vec_format('/home/sakakini/adr-detection-parent/large-files/word_vectors/PubMed-and-PMC-w2v.bin', binary=True)
infile_path_pos = "/home/sakakini/adr-detection-parent/large-files/datasets/craft-2.0/seq2seq-versions/pos-tagging/pos-tagging.txt"
infile_path_ner = "/home/sakakini/adr-detection-parent/large-files/datasets/ADE/ADE_NER_All.txt"
latent_dim = 256
batch_size = 1000
epochs = 300
n_instances = 100000
char_emb_size = 25
window_size = 7

(source_sents_ner, target_sents_ner) = read_data(infile_path_ner)
(source_sents_pos, target_sents_pos) = read_data(infile_path_pos)

rate = int(len(source_sents_pos)/len(source_sents_ner))

(source_sents_ner, target_sents_ner) = upsample(source_sents_ner, target_sents_ner, rate)

min_length = min([len(source_sents_ner), len(source_sents_pos)])
n_instances = min([n_instances, min_length])

source_sents_ner = source_sents_ner[:n_instances]
target_sents_ner = target_sents_ner[:n_instances]

source_sents_pos = source_sents_pos[:n_instances]
target_sents_pos = target_sents_pos[:n_instances]

source_sents = source_sents_ner + source_sents_pos
target_sents = target_sents_ner + target_sents_pos

source_vocab = collect_vocab(source_sents)
target_vocab = collect_vocab(target_sents)

target_vocab_pos = collect_vocab(target_sents_pos)
target_vocab_ner = collect_vocab(target_sents_ner)

target_vocab_size_pos = len(target_vocab_pos.keys())
target_vocab_size_ner = len(target_vocab_ner.keys())

source_vocab_size = len(source_vocab.keys())
target_vocab_size = len(target_vocab.keys())

(max_length_pos, source_sents_ints_pos, target_sents_ints_delayed_pos, target_sents_1hot_pos) = matricize_data(source_sents_pos, target_sents_pos, source_vocab, target_vocab, target_vocab_pos)
(max_length_ner, source_sents_ints_ner, target_sents_ints_delayed_ner, target_sents_1hot_ner) = matricize_data(source_sents_ner, target_sents_ner, source_vocab, target_vocab, target_vocab_ner)

emb_matrix = prepare_embedding_matrix(source_vocab)

model_input_source_pos = Input(shape = (max_length_pos,), dtype='int32')
model_input_target_pos = Input(shape = (max_length_pos,), dtype='int32')
model_input_source_ner = Input(shape = (max_length_ner,), dtype='int32')
model_input_target_ner = Input(shape = (max_length_ner,), dtype='int32')

embedding_layer_fixed = Embedding(source_vocab_size, 200, weights = [emb_matrix], trainable=False)
embedding_layer_trainable = Embedding(input_dim = source_vocab_size, output_dim = 300)
embedding_layer_target = Embedding(output_dim=30, input_dim=target_vocab_size)

source_embedding_fixed_pos = embedding_layer_fixed(model_input_source_pos)
source_embedding_trainable_pos = embedding_layer_trainable(model_input_source_pos)
target_embedding_pos = embedding_layer_target(model_input_target_pos)
source_embedding_fixed_ner = embedding_layer_fixed(model_input_source_ner)
source_embedding_trainable_ner = embedding_layer_trainable(model_input_source_ner)
target_embedding_ner = embedding_layer_target(model_input_target_ner)

dropout_layer_1 = Dropout(0.9)
LSTM_layer= Bidirectional(LSTM(latent_dim, return_sequences = True))

lstm_input_pos = concatenate([source_embedding_fixed_pos, source_embedding_trainable_pos])
lstm_input_pos = dropout_layer_1(lstm_input_pos)
lstm_output_pos = LSTM_layer(lstm_input_pos)
lstm_input_ner = concatenate([source_embedding_fixed_ner, source_embedding_trainable_ner])
lstm_input_ner = dropout_layer_1(lstm_input_ner)
lstm_output_ner = LSTM_layer(lstm_input_ner)

dropout_layer_2 = Dropout(0.9)
dense_layer_pos = Dense(target_vocab_size_pos, activation='softmax')
dense_layer_ner = Dense(target_vocab_size_ner, activation='softmax')

dense_input_pos = concatenate([lstm_output_pos, target_embedding_pos])
dense_input_pos = dropout_layer_2(dense_input_pos)
prediction_pos = dense_layer_pos(dense_input_pos)
dense_input_ner = concatenate([lstm_output_ner, target_embedding_ner])
dense_input_ner = dropout_layer_2(dense_input_ner)
prediction_ner = dense_layer_ner(dense_input_ner)

full_model = Model(inputs = [model_input_source_pos, model_input_target_pos, model_input_source_ner, model_input_target_ner], outputs = [prediction_pos, prediction_ner])


# Define Bidirectional models

bi_model_pos = Model(inputs = [model_input_source_pos], outputs = [lstm_output_pos])
bi_model_ner = Model(inputs = [model_input_source_ner], outputs = [lstm_output_ner])
#model.save("bidirectional.h5")

# Define Prediction model - NER

lstm_state_ner = Input(shape = (1,latent_dim*2,), dtype="float32")
previous_tag_ner = Input(shape = (1,), dtype="int32")
previous_tag_embedding_ner = embedding_layer_target(previous_tag_ner)
decoder_dense_input_ner = concatenate([lstm_state_ner, previous_tag_embedding_ner])
decoder_prediction_ner = dense_layer_ner(decoder_dense_input_ner)
dense_model_ner = Model(inputs = [lstm_state_ner, previous_tag_ner], outputs = [decoder_prediction_ner])
#model.save("prediction_model.h5") 

# Define Prediction model - POS

lstm_state_pos = Input(shape = (1,latent_dim*2,), dtype="float32")
previous_tag_pos = Input(shape = (1,), dtype="int32")
previous_tag_embedding_pos = embedding_layer_target(previous_tag_pos)
decoder_dense_input_pos = concatenate([lstm_state_pos, previous_tag_embedding_pos])
decoder_prediction_pos = dense_layer_pos(decoder_dense_input_pos)
dense_model_pos = Model(inputs = [lstm_state_pos, previous_tag_pos], outputs = [decoder_prediction_pos])
#model.save("prediction_model.h5") 

gpu_model = multi_gpu_model(full_model, gpus=get_available_gpus())
optimizer = RMSprop(lr=0.001)
gpu_model.compile(optimizer=optimizer, loss='categorical_crossentropy', loss_weights=[0.2, 1.0])
cbk_full = MyCbk_full(full_model)
cbk_bi_pos = MyCbk_bi_pos(bi_model_pos)
cbk_bi_ner = MyCbk_bi_ner(bi_model_ner)
cbk_dense_pos = MyCbk_dense_pos(dense_model_pos)
cbk_dense_ner = MyCbk_dense_ner(dense_model_ner)
#gpu_model.fit([np.array(source_sents_ints), np.array(target_sents_ints_delayed)], [np.array(target_sents_1hot)], batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[cbk_full, cbk_bi, cbk_dense])
gpu_model.fit([np.array(source_sents_ints_pos), np.array(target_sents_ints_delayed_pos), np.array(source_sents_ints_ner), np.array(target_sents_ints_delayed_ner)], [np.array(target_sents_1hot_pos), np.array(target_sents_1hot_ner)], batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[cbk_full, cbk_bi_pos, cbk_bi_ner, cbk_dense_pos, cbk_dense_ner])
