from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Dropout
from keras.utils import to_categorical, multi_gpu_model
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
import keras.backend as K
import numpy as np

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

"""
def sent2tags(sent_index):
	pred = model.predict(np.reshape(source_sents_ints[sent_index],(1,79)))
	pred_seq = []
	for i in range(79):
		arg = np.argmax(pred[0,i,:])
		pred_seq.append(target_index2word[arg])
	return pred_seq
"""

def sent2tags(sent_index):
	input_ints = np.array(source_sents_ints[sent_index]).reshape(1, max_length)
	lstm_states = bidirectional_model.predict(input_ints)
	tag_int = np.array([target_vocab["<s>"]])
	tag_seq = []
	for i in range(max_length):
		lstm_state = lstm_states[:,i,:].reshape(1,1,latent_dim*2)
		output = prediction_model.predict([lstm_state, tag_int])
		tag_ind = np.argmax(output)
		tag_int = np.array([tag_ind])
		tag = target_index2word[tag_ind]
		tag_seq.append(tag)
		if tag == "<PAD>":
			break
	return tag_seq
		
		





	"""
	char_input_ints = source_char_ints[sent_index]
	h = np.zeros((1,latent_dim), dtype = "float32")
	c = np.zeros((1,latent_dim), dtype = "float32")
	tag_int = np.array([target_vocab["<s>"]])
	tag_seq = []
	for i in range(len(input_ints)):
		word_int = np.array([input_ints[i]])
		chars_int = np.array([char_input_ints[i]]).reshape((1,1,max_word_len))
		output, h, c = decoder_model.predict([word_int, tag_int, chars_int, h, c])
		tag_ind = np.argmax(output)
		tag_int = np.array([tag_ind])
		tag = target_index2word[tag_ind]
		tag_seq.append(tag)
		if tag == "<PAD>":
			break
	return tag_seq
	"""
	 
def evaluate_full(indices_range, tag_list):

	TP = 0.0
	FP = 0.0
	FN = 0.0
	eps = 0.00001
	for ind in indices_range:
		predicted_seq = sent2tags(ind)
		gold_seq = target_sents[ind][1:]
		predicted_ents = seq2ent(predicted_seq)
		gold_ents = seq2ent(gold_seq)
		TP += len(gold_ents & predicted_ents)
		FP += len(gold_ents - predicted_ents)
		FN += len(predicted_ents - gold_ents)
	P = TP/(TP+FP+eps)
	R = TP/(TP+FN+eps)
	F = (2*P*R)/(P+R+eps)
	print(P,R,F)

def evaluate(indices_range, tag_list):
	TP = 0.0
	FP = 0.0
	FN = 0.0
	eps = 0.00001
	for ind in indices_range:
		predicted_seq = sent2tags(ind)
		gold_seq = target_sents[ind][1:]
		#print("Predicted Sequence:", predicted_seq)
		#print("Gold Sequence:", gold_seq)
		#print("Predicted ents:", seq2ent(predicted_seq))
		#print("Gold ents:", seq2ent(gold_seq))
		#print("Scores before (TP,FP,FN)", TP, FP, FN)
		for tag in tag_list:
			tag_gold = set([t for t,j in enumerate(gold_seq) if j == tag])
			tag_pred = set([t for t,j in enumerate(predicted_seq) if j == tag])
			TP += len(tag_gold & tag_pred)
			FP += len(tag_gold - tag_pred)
			FN += len(tag_pred - tag_gold)
		#print("Scores after (TP,FP,FN)", TP, FP, FN)
	P = TP/(TP+FP+eps)
	R = TP/(TP+FN+eps)
	F = (2*P*R)/(P+R+eps)
	print(P,R,F)

def seq2ent(seq):
	ents = []
	ent_start_flag = False
	ent_type = None
	ent_start = 0
	ent_end = 0
	for i,tag in enumerate(seq):
		if tag[0] == "B":
			ent_start = i
			ent_type = tag[-1]
			ent_start_flag = True
		elif tag == "O" and ent_start_flag:
			ents.append((ent_type,ent_start,i))
			ent_start_flag = False
	return set(ents)
	
def read_data(infile_path):
        infile = open(infile_path)
        source_sents = []
        target_sents = []
        for line in infile:
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

infile_path = "/home/sakakini/adr-detection-parent/large-files/datasets/craft-2.0/seq2seq-versions/pos-tagging/pos-tagging.txt"
latent_dim = 256
batch_size = 1000
epochs = 10

(source_sents, target_sents) = read_data(infile_path)
source_vocab = collect_vocab(source_sents)
target_vocab = collect_vocab(target_sents)
source_sents_ints = sents2ints(source_sents, source_vocab)
target_sents_ints_delayed = sents2ints(target_sents, target_vocab)

(source_char_ints, max_word_len, char_vocab) = word_ints_2_char_ints(source_sents_ints, source_vocab)

target_sents_ints = [sent[1:] for sent in target_sents_ints_delayed]
target_sents_ints_delayed = [sent[:-1] for sent in target_sents_ints_delayed]

target_index2word = dict((index, word) for word, index in target_vocab.items())

source_vocab_size = len(source_vocab.keys())
target_vocab_size = len(target_vocab.keys())

max_length = max([len(x) for x in target_sents_ints])

target_sents_1hot = ints21hot(target_sents_ints, target_vocab_size + 1, max_length)

#decoder_model = load_model("decoder.h5")
bidirectional_model = load_model("bi.h5")
prediction_model = load_model("dense.h5")

tag_list = list(target_vocab.keys())
tag_list.remove('<PAD>')

train_size = int(0.8*len(source_sents))
test_size = len(source_sents) - train_size

test_size = 500

print(train_size, test_size)

print("Evaluation data scores:")
evaluate(range(train_size,train_size+test_size), tag_list)
print("Training data scores:")
evaluate(range(0,train_size), tag_list)
