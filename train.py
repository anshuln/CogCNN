from model import MultiTaskModel
from tqdm import tqdm

import tensorflow as tf
import numpy as np

import os


train_dir = 'data/'
label_im_dir = 'label/images'
label_dir = 'label/labels'

num_inputs = 2
image_shape = (300,300,3)
num_labels = 31

def train(model,optimizer,epochs):	#TODO add validation, generator for datasets
	files = []
	for file in os.listdir(train_dir):
	    if file.endswith(".npy"):	#Change to h5 later
	    	files.append(file)
	for epoch in epochs:
		loss = []
		for f in tqdm(files):
			input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
			label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
			label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
			l = model.train_on_batch(input_batch,label_im_batch,label_batch,optimizer)
			loss.append(l.numpy())
		print("Epoch - {} Loss - {}".format(epoch,np.array(loss).mean()))

if __name__ == "__main__":
	TF_CPP_MIN_LOG_LEVEL = 3
	model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels)
	d1 = np.load("{}/0.npy".format(train_dir),allow_pickle=True)
	pred = model.call(d1)
	optimizer = tf.optimizers.Adam()
	model.setTrainableVariables()
	train(model,optimizer,5)

