from model import MultiTaskModel
from tqdm import tqdm

import tensorflow as tf
import numpy as np

import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
train_dir = 'data/'
label_im_dir = 'label/images'
label_dir = 'label/labels'
log_file = 'log.txt'

num_inputs = 4
image_shape = (64,64,3)
num_labels = 31


def train(model,optimizer,epochs):  #TODO add validation, generator for datasets
	train_files = []
	val_files = []
	for file in os.listdir(train_dir):
		if file.endswith(".npy"):   #Change to h5 later
			train_files.append(file)
	for file in os.listdir(train_dir):
		if file.endswith("v.npy"):   #Change to h5 later
			val_files.append(file)
	train_files = [x for x in train_files if x not in val_files]
	log = open(log_file,"w")
	for epoch in range(epochs):
		print("Epoch num {}".format(epoch))
		tloss = []
		vloss = []
		v_acc = 0
		for f in tqdm(train_files):
			input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
			label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
			label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
			l,l_all = model.train_on_batch(input_batch,label_im_batch,label_batch,optimizer)
			tloss.append([l.numpy() for l in l_all])
		for f in val_files:
			input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
			label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
			label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
			l,l_all = model.validate_batch(input_batch,label_im_batch,label_batch)
			v_acc += l 
			vloss.append([l.numpy() for l in l_all])

		log.write("Epoch - {} Training_Loss - {} validation_loss - {}, v_acc = {}".format(epoch,np.array(tloss).mean(axis=0),np.array(vloss).mean(axis=0),v_acc/(16*len(val_files))))
		log.write("\n")
		print("Val-loss{}".format(np.array(vloss).mean(axis=0)[-1]))
		if epoch % 5 == 0:
			pickle.dump(model.trainableVariables,open("trained_models/model_{}.pkl".format(epoch),"wb"))

if __name__ == "__main__":
	TF_CPP_MIN_LOG_LEVEL = 4
	model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels)
	d1 = np.load("{}/0.npy".format(train_dir),allow_pickle=True)
	print(d1[:,:1].shape)
	print("Building Model...")
	pred = model.call(d1[:,:1])
	for x in pred:
		print("Shape",x.shape)  
	print("Model built")
	optimizer = tf.optimizers.Adam()
	model.setTrainableVariables()
	train(model,optimizer,1000)
	# pred = model.call(d1)



