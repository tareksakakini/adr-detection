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
	input_ints = source_sents_ints[sent_index]
	h = np.zeros((1,latent_dim), dtype = "float32")
	c = np.zeros((1,latent_dim), dtype = "float32")
	tag_int = np.array([target_vocab["<s>"]])
	tag_seq = []
	for i in range(len(input_ints)):
		word_int = np.array([input_ints[i]])
		output, h, c = decoder_model.predict([word_int, tag_int, h, c])
		tag_ind = np.argmax(output)
		tag_int = np.array([tag_ind])
		tag = target_index2word[tag_ind]
		tag_seq.append(tag)
		if tag == "<PAD>":
			break
	return tag_seq
	 


def evaluate(indices_range):
	TP = 0.0
	FP = 0.0
	FN = 0.0
	eps = 0.00001
	for ind in indices_range:
		predicted_seq = sent2tags(ind)
		gold_seq = target_sents[ind][1:]
		B_M_gold = set([t for t,j in enumerate(gold_seq) if j == "B-M"])
		I_M_gold = set([t for t,j in enumerate(gold_seq) if j == "I-M"])
		B_M_pred = set([t for t,j in enumerate(predicted_seq) if j == "B-M"])
		I_M_pred = set([t for t,j in enumerate(predicted_seq) if j == "I-M"])
		TP += (len(B_M_gold & B_M_pred) + len(I_M_gold & I_M_pred))
		FP += (len(B_M_gold - B_M_pred) + len(I_M_gold - I_M_pred))
		FN += (len(B_M_pred - B_M_gold) + len(I_M_pred - I_M_gold))
	P = TP/(TP+FP+eps)
	R = TP/(TP+FN+eps)
	F = (2*P*R)/(P+R+eps)
	print(P,R,F)
	
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


infile_path = "/home/sakakini/adr-detection-parent/large-files/datasets/ADE_NER_Meds.txt"
latent_dim = 256
batch_size = 1000
epochs = 10

(source_sents, target_sents) = read_data(infile_path)
source_vocab = collect_vocab(source_sents)
target_vocab = collect_vocab(target_sents)
source_sents_ints = sents2ints(source_sents, source_vocab)
target_sents_ints_delayed = sents2ints(target_sents, target_vocab)

target_sents_ints = [sent[1:] for sent in target_sents_ints_delayed]
target_sents_ints_delayed = [sent[:-1] for sent in target_sents_ints_delayed]

target_index2word = dict((index, word) for word, index in target_vocab.items())

source_vocab_size = len(source_vocab.keys())
target_vocab_size = len(target_vocab.keys())

max_length = max([len(x) for x in target_sents_ints])

target_sents_1hot = ints21hot(target_sents_ints, target_vocab_size + 1, max_length)

decoder_model = load_model("decoder.h5")


print("Evaluation data scores:")
evaluate(range(4981,6227))
print("Training data scores:")
evaluate(range(0,4981))
