from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Dropout, concatenate
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
		
wv_model = KeyedVectors.load_word2vec_format('/home/sakakini/adr-detection/word_vectors/PubMed-and-PMC-w2v.bin', binary=True)
infile_path = "/home/sakakini/adr-detection/datasets/ADE_NER_Meds.txt"
latent_dim = 256
batch_size = 1000
epochs = 300
n_instances = 10000

(source_sents, target_sents) = read_data(infile_path)
source_sents = source_sents[:n_instances]
target_sents = target_sents[:n_instances]
source_vocab = collect_vocab(source_sents)
target_vocab = collect_vocab(target_sents)
source_sents_ints = sents2ints(source_sents, source_vocab)
target_sents_ints_delayed = sents2ints(target_sents, target_vocab)

target_sents_ints = [sent[1:] for sent in target_sents_ints_delayed]
target_sents_ints_delayed = [sent[:-1] for sent in target_sents_ints_delayed]

source_vocab_size = len(source_vocab.keys())
target_vocab_size = len(target_vocab.keys())

emb_matrix = prepare_embedding_matrix(source_vocab)

max_length = max([len(x) for x in target_sents_ints])

target_sents_1hot = ints21hot(target_sents_ints, target_vocab_size, max_length)

model_input_source = Input(shape = (max_length,), dtype='int32')
model_input_target = Input(shape = (max_length,), dtype='int32')
#embedding_layer_trainable = Embedding(output_dim=200, input_dim=source_vocab_size)
embedding_layer_fixed = Embedding(source_vocab_size, 200, weights = [emb_matrix], trainable=False)
embedding_layer_target = Embedding(output_dim=30, input_dim=target_vocab_size)
source_embedding_fixed = embedding_layer_fixed(model_input_source)
target_embedding = embedding_layer_target(model_input_target)
#source_embedding_trainable = embedding_layer_trainable(model_input)
#source_embedding = concatenate([source_embedding_fixed, source_embedding_trainable])
lstm_input = concatenate([source_embedding_fixed, target_embedding])
LSTM_layer= LSTM(latent_dim, return_sequences = True, return_state = True)
lstm_output, h, c = LSTM_layer(lstm_input)
dense_layer = Dense(target_vocab_size, activation='softmax')
prediction = dense_layer(lstm_output)
model = Model(inputs = [model_input_source, model_input_target], outputs = [prediction])
#model = multi_gpu_model(model, gpus=2)
optimizer = RMSprop(lr=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.fit([np.array(source_sents_ints), np.array(target_sents_ints_delayed)], [np.array(target_sents_1hot)], batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save("model.h5")

# defining prediction module

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_word_input = Input(shape=(1,))
decoder_tag_input = Input(shape=(1,))
decoder_word_embedding = embedding_layer_fixed(decoder_word_input)
decoder_tag_embedding = embedding_layer_target(decoder_tag_input)
decoder_lstm_input = concatenate([decoder_word_embedding,decoder_tag_embedding])
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = LSTM_layer(decoder_lstm_input, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = dense_layer(decoder_outputs)
decoder_model = Model([decoder_word_input, decoder_tag_input] + decoder_states_inputs,[decoder_outputs] + decoder_states)
decoder_model.save("decoder.h5")
