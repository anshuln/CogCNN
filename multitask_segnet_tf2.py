from  tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,ReLU,BatchNormalization,Activation
from layers import *
import pickle
# import argparse 

# parser = argparse.ArgumentParser(description='Multi-task: Split')
# parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
# parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
# parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
# parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
# opt = parser.parse_args()

class blocks(tf.keras.Sequential):
	def __init__(self):
		super(blocks,self).__init__()
	def conv_layer(self, channel):
		conv_block = tf.keras.Sequential(
			[Conv2D(filters=channel, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),
			BatchNormalization(axis=-1),
			ReLU()]
		)
		return conv_block

class encoder(blocks):
	def __init__(self,channels=3):
		super(encoder,self).__init__()
		filter = [64, 128, 256, 512, 512]
		self.conv_block_enc = []
		self.conv_block_enc.append(Sequential([self.conv_layer(filter[0]),self.conv_layer(filter[0])]))
		for i in range(4):  #TODO Refactor for better model making
			if i == 0:
				self.conv_block_enc.append(Sequential([self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1])]))
			else:
				self.conv_block_enc.append(Sequential([self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1]),
													self.conv_layer(filter[i + 1])]))
		self.down_sampling = MaxPool2D(ksize=(2,2),padding='same')
	def call(self,x):
		x1 = x
		indices = []
		sizes = []
		for i in range(5):
			x1 = self.conv_block_enc[i](x1)
			sizes.append(x1.shape)
			x1,index = self.down_sampling.call(x1)
			indices.append(index)
			#print("Encode",x1.shape,indices[-1].shape)
		encout = x1
		indices = indices
		return x1,indices,sizes

class decoder(blocks):
	def __init__(self,channels=3):
		super(decoder,self).__init__()
		filter = [64, 128, 256, 512, 512]
		self.conv_block_dec = []
#       self.conv_block_dec = Sequential()
		for i in range(1,4):
			self.conv_block_dec.append(Sequential([self.conv_layer(filter[-i]),
												  self.conv_layer(filter[-i]),
												  self.conv_layer(filter[-(i+1)])]))

		self.conv_block_dec.append(Sequential([self.conv_layer(filter[1]),
												  self.conv_layer(filter[0])]))
		self.conv_block_dec.append(Sequential([self.conv_layer(filter[0]),
												  tf.keras.Sequential(
			[BatchNormalization(axis=-1),
			Conv2D(filters=channels, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid"),]
		)]))    #Getting best results when sigmoid, batch_norm, relu
		
		self.up_sampling = MaxUnpool2D(ksize=(2,2))
	def forward(self,X,indices,sizes):
		indices = indices[::-1]
		sizes = sizes[::-1]
		for idx,layer in enumerate(self.conv_block_dec):
			#print(X.shape,indices[idx].shape)
			# print(idx,X.shape,self.max_indices[idx].shape)
			X = self.up_sampling.call([X,indices[idx]],sizes[idx])
			#print(idx,X.shape,indices[idx].shape)
			X = layer(X) 
		return X

class SegNet(tf.keras.Sequential):
	def __init__(self):
#       encoder is a model of type encoder defined above,
#       decoders is a list of decoders 
		super(SegNet,self).__init__()
		self.encoder = encoder()
		self.decoder = decoder()
		self.TrainableVarsSet = False
		self.optimizer = tf.keras.optimizers.Adam()
	def setTrainableVars(self):
		self.TrainableVars = self.encoder.trainable_variables+self.decoder.trainable_variables
		self.TrainableVarsSet = True
	def call(self,X):
		X,indices,sizes = self.encoder.call(X)
		return X,self.decoder.forward(X,indices,sizes)
	def save(self,fileName):
		pickle.dump(self.encoder.get_weights(),open("{}-enc".format(fileName),"wb"))
		pickle.dump(self.decoder.get_weights(),open("{}-dec".format(fileName),"wb"))

	def load_model(self,fileName):
		enc_train_vars = pickle.load(open("{}-enc".format(fileName),"rb"))
		dec_train_vars = pickle.load(open("{}-dec".format(fileName),"rb"))
		for l in self.encoder.layers:
			# weights = l.get_weights()
			weights = enc_train_vars
			l.set_weights(weights)
		for l in self.decoder.layers:
			# weights = l.get_weights()
			weights = dec_train_vars
			l.set_weights(weights)
		self.TrainableVarsSet = False
