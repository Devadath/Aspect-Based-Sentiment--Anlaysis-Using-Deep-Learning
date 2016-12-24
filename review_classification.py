#!/usr/bin/python
import numpy as np
import json
import os
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

from sklearn.metrics import accuracy_score

train_file = "/home/devan/Documents/aspects-annotated-dataset/tripadvisor/train.unique.json"
test_file  = "/home/devan/Documents/aspects-annotated-dataset/tripadvisor/test.unique.json"

def extract_data_json(file):
	"""
	This function reads the .json file line by line
	"""
	data = []
	for line in open(file):
		data.append(json.loads(line))
	return data

def take_data(file):
	"""
	This function extracs the segments and segment lables from the jason files and make a key:value pair 
	of segments and segment labels. This returns an ordered dictionary of "segments":"segment labels" pairs.
	"""
	segs, seglabel= [], []
	for line in file:
		segs += line['segments']
		seglabel += [ i for i in line["segmentLabels"] ]
	annotation = OrderedDict(zip(segs,seglabel))
	return annotation

def make_trainable(ann_data):
	"""
	This makes a list of sentences and a list of feature aspects + opinion.
	"""
	text, labels = [], []
	for item in ann_data.items():
		text.append(item[0])
		label_dicts = item[1]
		extra = []
		for i in label_dicts:
			extra.append(i+"_"+label_dicts[i])
		labels.append(extra)
	return text, labels

def to_categorical(lab_list):
	"""
	Converts labels to 1 of k encoding.
	"""
	y_cat = []
	labels = ['BUILDING_in', 'BUILDING_ip', 'BUILDING_ix', 'BUILDING_n', 'BUILDING_p', 'BUILDING_x', 'BUSINESS_in', 'BUSINESS_ip', 'BUSINESS_p', 'BUSINESS_x', 'CHECKIN_in', 'CHECKIN_ip', 'CHECKIN_ix', 'CHECKIN_n', 'CHECKIN_p', 'CHECKIN_x', 'CLEANLINESS_in', 'CLEANLINESS_ip', 'CLEANLINESS_n', 'CLEANLINESS_p', 'CLEANLINESS_x', 'FOOD_i', 'FOOD_in', 'FOOD_ip', 'FOOD_ix', 'FOOD_n', 'FOOD_p', 'FOOD_x', 'LOCATION_in', 'LOCATION_ip', 'LOCATION_ix', 'LOCATION_n', 'LOCATION_p', 'LOCATION_x', 'NOTRELATED_in', 'NOTRELATED_ip', 'NOTRELATED_n', 'NOTRELATED_p', 'NOTRELATED_x', 'OTHER_in', 'OTHER_ip', 'OTHER_ix', 'OTHER_n', 'OTHER_p', 'OTHER_x', 'ROOMS_in', 'ROOMS_ip', 'ROOMS_ix', 'ROOMS_n', 'ROOMS_p', 'ROOMS_x', 'SERVICE_in', 'SERVICE_ip', 'SERVICE_ix', 'SERVICE_n', 'SERVICE_p', 'SERVICE_x', 'VALUE_in', 'VALUE_ip', 'VALUE_n', 'VALUE_p', 'VALUE_x']
	for i in lab_list:
		cat = [0.]*63
		for j in i:
			if j in labels:
				cat[labels.index(j)] = 1
		y_cat.append(cat)
	return np.array(y_cat)


#Preprocessing training data
train_data = extract_data_json(train_file)
train_datalabels = take_data(train_data)
train_text, labels = make_trainable(train_datalabels)
train_labels = to_categorical(labels)


#Preprocessig test data
test_data = extract_data_json(test_file)
test_datalabels = take_data(test_data)
test_text, labels = make_trainable(test_datalabels)
test_labels = to_categorical(labels)

#Tokenising the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
tokenizer.fit_on_texts(test_text)

#Converting text to sequences
train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

#padding
train_data = pad_sequences(train_sequences, maxlen=40)
test_data = pad_sequences(test_sequences, maxlen=40)


#Deciding the validation data
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
data = train_data[indices]
labels = train_labels[indices]
_validation_samples = int(0.2 * data.shape[0])

#Splitting training data to training data and validation data
x_train = data[:-_validation_samples]
y_train = labels[:-_validation_samples]
x_val = data[-_validation_samples:]
y_val = labels[-_validation_samples:]

print("Training data:", x_train.shape, "Training labels:", y_train.shape, "Validation data:", x_val.shape, "Validation_labels", y_val.shape)
print("Test data : ", test_data.shape, "Test_labels:", test_labels.shape)


print('Building model...')
model = Sequential()

#Embedding layer
model.add(Embedding(10000, 128, dropout=0.2))

#lstm
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 

#final layer
model.add(Dense(63))
model.add(Activation('sigmoid'))

#compiling
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(' Starting Training.......')
model.fit(x_train, y_train, batch_size=32, nb_epoch=15, validation_data=(x_val, y_val))

#Evaluation on validation data
error_loss, accuracy = model.evaluate(x_val, y_val, batch_size=32)

print('Validation loss:', error_loss, 'Test accuracy:', accuracy)


#prediction on test data
preds = model.predict(test_data)

#probability to one of k.
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

predictions = [x for x in preds]

#Calculating overall Accuracy.
score = float(0)	
for i, j in zip(test_labels, predictions):
	ind_accuracy = accuracy_score(i, j, normalize=False)
	score = score+(ind_accuracy/63)
accuracy = float(score/1484*63)

print( 'Overall Accuracy:', str(accuracy)+"%")	