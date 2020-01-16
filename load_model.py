import tensorflow as tf 
import numpy as np 
from model_bits_and_pieces import MultiTaskModel
import os
import matplotlib.pyplot as plt
num_inputs = 4
image_shape = (64,64,3)
num_labels = 31
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
	tf.config.experimental.set_memory_growth(d, True)

if __name__ == '__main__':
	model = MultiTaskModel(num_labels=num_labels,num_inputs=num_inputs,image_shape=image_shape,attention="multi",two_stage=True,pix2pix=False)
	train_dir = 'data/'
	d1 = np.load("{}/0v.npy".format(train_dir),allow_pickle=True)
	model.build(d1[:,:1])
	model.load_model("../results(8)",attention="multi",two_stage=True,pix2pix=False)
	a = (model.getAttentionMap(d1[:,:1])[0][0])
	# fig,ax = plt.subplots(1,2)
	# ax[0].imshow(a.reshape((256,32)))
	# ax[1].imshow(a[:,:,:512].reshape((64,32)))
	# plt.show()
	for i in range(8):
		aten_rec,aten_pred,prediction = model.getAttentionMap(d1[:,i:i+1])
		print(i)
		fig, ax = plt.subplots(2,4)
		ax[0,0].imshow(aten_rec[:,:,:,:512].reshape((64,32)))
		ax[0,1].imshow(aten_rec[:,:,:,512:1024].reshape((64,32)))
		ax[0,2].imshow(aten_rec[:,:,:,1024:1536].reshape((64,32)))
		ax[0,3].imshow(aten_rec[:,:,:,1536:2048].reshape((64,32)))
		ax[1,0].imshow(aten_pred[:,:,:,:512].reshape((64,32)))
		ax[1,1].imshow(aten_pred[:,:,:,512:1024].reshape((64,32)))
		ax[1,2].imshow(aten_pred[:,:,:,1024:1536].reshape((64,32)))
		ax[1,3].imshow(aten_pred[:,:,:,1536:2048].reshape((64,32)))
		plt.savefig("AttentionMaps/new-{}.png".format(i))
		plt.close()
