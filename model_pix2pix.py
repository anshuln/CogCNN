#TODO Rename file
import tensorflow as tf
from multitask_segnet_tf2 import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Reshape,Dropout,BatchNormalization,Conv2D,ZeroPadding2D,LeakyReLU
import numpy as np
import pickle
class PatchGanDiscriminator(tf.keras.Model):
	def downsample(filters, size, apply_batchnorm=True):
		initializer = tf.random_normal_initializer(0., 0.02)

		result = tf.keras.Sequential()
		result.add(Conv2D(filters, size, strides=2, padding='same',
								 kernel_initializer=initializer, use_bias=False))

		if apply_batchnorm:
			result.add(tf.keras.layers.BatchNormalization())

		result.add(tf.keras.layers.LeakyReLU())

		return result
	def __init__(self):
		super(PatchGanDiscriminator, self).__init__()
		# self.down1 = self.downsample(64, 4, False) # (bs, 128, 128, 64)
		# self.down2 = self.downsample(128, 4) # (bs, 64, 64, 128)
		self.down = self.downsample(256, 4) # (bs, 32, 32, 256)
		self.conv1 = Conv2D(512, 4, strides=1,
								use_bias=False)
		self.conv2 = Conv2D(1, 4, strides=1,
								use_bias=False)

	def call(self,X_inp,X_gen):
		X = tf.concat([X_inp,X_gen],axis=-1)
		X = self.down.call(X)
		X = ZeroPadding2D()(X)
		X = self.conv1.call(X)
		X = BatchNormalization()(X)
		X = LeakyReLU()(X)
		X = ZeroPadding2D()(X)
		X = self.conv2.call(X)
		return X

class MultiTaskModel(Sequential):
	def __init__(self,image_shape,num_labels,num_inputs=4,trainableVariables=None):
		#num_inputs refers to input channels(edge,texture etc.)
		#image_shape is the shape of 1 image for reconstruction
		#TODO - kwargs support for segnet initializations
		super(MultiTaskModel, self).__init__()
		self.num_inputs = num_inputs    
		self.image_shape = image_shape
		self.segnets = []
		if trainableVariables is None:
			self.trainableVariables = []    #Not to be confused with trainable_variables, which is read-only
		else:
			self.trainableVariables = trainableVariables
		for i in range(num_inputs):
			self.segnets.append(SegNet())
		print("Image_Shape",image_shape)
		self.reconstruct_image = Sequential([Flatten(),Dense(1000),BatchNormalization(axis=-1)
				,Dense(image_shape[0]*image_shape[1]*image_shape[2],activation='sigmoid')])
		#Uncomment the two lines below to enable classification
		self.predict_label = Sequential([Flatten(),Dense(1000),BatchNormalization(axis=-1),
				Dense(num_labels,activation='softmax')])    #The loss function uses softmax, final preds as well
		# self.discriminator = PatchGanDiscriminator()
		# self.discriminator.compile()

	def setTrainableVariables(self,trainableVariables=None):
		if trainableVariables is not None:
			self.trainableVariables = trainableVariables
			return
		for i in range(self.num_inputs):    
			print("On segnet",i)
			self.trainableVariables += self.segnets[i].trainable_variables
		self.trainableVariables += self.reconstruct_image.trainable_variables
		#Uncomment the two lines below to enable classification
		self.trainableVariables += self.predict_label.trainable_variables 
	
	@tf.function
	def call(self,X):
		#X is a LIST of the dimension [batch*h*w*c]*num_inputs
		#TODO check if this gives us correct appending upon flatten
		#TODO refactor to make everything a tensor
		batch,h,w,c = X[0].shape
		# print("X.shape",h,w,c)
		assert len(X) == self.num_inputs
		result = []
		encoded_reps,rec = self.segnets[0].call(X[0])
		encoded_reps = tf.expand_dims(encoded_reps,1)
		result.append(rec)
		for i in range(self.num_inputs-1):
			enc,rec = self.segnets[i+1].call(X[i+1])
			enc = tf.expand_dims(enc,1)
			encoded_reps = tf.concat([encoded_reps,enc],axis=1)
			result.append(rec)  #Appending the reconstructed result to return 
		#print(encoded_reps.shape)
		# print("Call_shape",encoded_reps.shape)
		result.append(tf.reshape(self.reconstruct_image(encoded_reps),(batch,h,w,c)))   #Appending final image
		#Uncomment the two lines below to enable classification
		result.append(self.predict_label(encoded_reps))     #Appending final labels
		result.append(encoded_reps)	#Needed for pix2pix
		return result

	def loss_reconstruction(self,X,Y,beta=0.0):
		# print(X.shape,Y.shape)
		#Pixel-wise l2 loss
		# return  tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum((X-Y)**2,
			# axis=-1),axis=-1),axis=-1,keepdims=True)    #see if keepdims is required
		return (1-beta)*tf.math.reduce_sum((X-Y)**2)/(X.shape[1]*X.shape[2]*X.shape[3]) + beta*tf.math.reduce_sum(tf.math.abs(X-Y))/(X.shape[1]*X.shape[2]*X.shape[3])

	def loss_classification(self,X,labels):
		return (-1*tf.reduce_mean(labels*(tf.math.log(X+1e-5)) + (1-labels)*(tf.math.log(1-X+1e-5))))

	def generator_loss(self,disc_generated_output, gen_output, target,LAMBDA=0.1):
		gan_loss = self.loss_classification(tf.ones_like(disc_generated_output), disc_generated_output)

		# mean absolute error
		l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

		total_gen_loss = gan_loss + (LAMBDA * l1_loss)

		return total_gen_loss, gan_loss, l1_loss
	def discriminator_loss(self,disc_real_output, disc_generated_output):
		real_loss = self.loss_classification(tf.ones_like(disc_real_output), disc_real_output)

		generated_loss = self.loss_classification(tf.zeros_like(disc_generated_output), disc_generated_output)

		total_disc_loss = real_loss + generated_loss

		return total_disc_loss



	def train_on_batch(self,X,Y_image,Y_labels,optimizer):
		# Y needs to be a list of [img,labels]
		with tf.GradientTape(persistent=True) as tape:
			
			result = self.call(X)
			losses = []
			loss = 0
			loss_disc = 0

			loss = self.loss_reconstruction(X[0],result[0])
			losses.append(loss)
			for i in range(self.num_inputs-1):
				loss += self.loss_reconstruction(X[i+1],result[i+1])
				losses.append(self.loss_reconstruction(X[i+1],result[i+1]))
			disc_real_output = self.discriminator.call(result[-1],Y_image)
			disc_generated_output = self.discriminator.call(result[-1],result[self.num_inputs])
			loss += self.generator_loss(disc_generated_output,result[self.num_inputs],Y_image)
			# losses.append(self.loss_reconstruction(result[self.num_inputs],Y_image))
			#Uncomment the two lines below to enable classification
			loss += self.loss_classification(result[self.num_inputs+1],Y_labels)
			losses.append(self.loss_classification(result[self.num_inputs+1],Y_labels))

			loss_disc += self.discriminator_loss(disc_real_output,disc_generated_output)
		grads = tape.gradient(loss,self.trainableVariables)
		grads_and_vars = zip(grads, self.trainableVariables)

		grad_disc = tape.gradient(loss_disc,self.discriminator.trainable_variables)
		grads_and_vars_disc = zip(grad_disc, self.discriminator.trainable_variables)

		optimizer.apply_gradients(grads_and_vars)
		optimizer.apply_gradients(grads_and_vars_disc)

		del tape
		return loss,losses

	def validate_batch(self,X,Y_image,Y_labels):
		# Returns predictions, losses on batch
		result = self.call(X)
		losses = []
		loss = self.loss_reconstruction(X[0],result[0])
		losses.append(loss)
		for i in range(self.num_inputs-1):
			loss += self.loss_reconstruction(X[i+1],result[i+1])
			losses.append(self.loss_reconstruction(X[i+1],result[i+1]))
		loss += self.loss_reconstruction(result[self.num_inputs],Y_image)
		# print("Loss: ",loss)
		losses.append(self.loss_reconstruction(result[self.num_inputs],Y_image))
		loss += self.loss_classification(result[self.num_inputs+1],Y_labels)
		losses.append(self.loss_classification(result[self.num_inputs+1],Y_labels))
		# print(result[-1].shape,Y_labels.shape,tf.math.argmax(result[-1],axis=1).numpy()==np.argmax(Y_labels,axis=1))
		return (np.array((tf.math.argmax(result[-1],axis=1).numpy()==np.argmax(Y_labels,axis=1)))*1.0).sum(),losses
		# return losses
		
	def getWeightNorms(self):
		#Returns ||W|| for each encoded rep
		weight_rec = self.reconstruct_image.layers[1].trainable_variables[0]
		# print("Norms",weight_rec.shape)
		weight_pred = self.predict_label.layers[1].trainable_variables[0]
		norms_rec = []
		norms_pred = []
		min_shape = weight_rec.shape[0]//self.num_inputs
		for i in range(self.num_inputs):
			w = weight_pred[i*min_shape:(i+1)*min_shape,:].numpy()
			norms_pred.append(np.sum(w**2))
			w = weight_rec[i*min_shape:(i+1)*min_shape,:].numpy()
			norms_rec.append(np.sum(w**2))
		return norms_rec,norms_pred

	def save(self,modelDir):
		for i in range(len(self.segnets)):
			self.segnets[i].save("{}/Segnet-{}".format(modelDir,i))
		pickle.dump(self.reconstruct_image.get_weights(),
			open("{}/Reconstruction-Model".format(modelDir),"wb"))
		pickle.dump(self.predict_label.get_weights(),
			open("{}/Prediction-Model".format(modelDir),"wb"))
		pickle.dump(self.discriminator.get_weights(),
			open("{}/Discriminator".format(modelDir),"wb"))

	def load_model(self,modelDir):
		for i in range(len(self.segnets)):
			self.segnets[i].load_model("{}/Segnet-{}".format(modelDir,i))
		rec_train_vars = pickle.load(open("{}/Reconstruction-Model".format(modelDir),"rb"))
		pred_train_vars = pickle.load(open("{}/Prediction-Model".format(modelDir),"rb"))
		disc_train_vars = pickle.load(open("{}/Discriminator".format(modelDir),"rb"))
		for l in self.reconstruct_image.layers:
			# weights = l.get_weights()
			weights = rec_train_vars
			l.set_weights(weights)
		for l in self.predict_label.layers:
			# weights = l.get_weights()
			weights = pred_train_vars
			l.set_weights(weights)
		for l in self.discriminator.layers:
			# weights = l.get_weights()
			weights = disc_train_vars
			l.set_weights(weights)
		self.TrainableVarsSet = False
