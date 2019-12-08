# This model does MTL in 2 stages, 
# First only the reconstruction tasks are trained
# Then encoder weights are fixed and classifier is trained 
import tensorflow as tf
from multitask_segnet_tf2 import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Reshape,Dropout,BatchNormalization
import numpy as np
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
	def call(self,X,classification=False):
		#X is a LIST of the dimension [batch*h*w*c]*num_inputs
		#TODO check if this gives us correct appending upon flatten
		#TODO refactor to make everything a tensor
		batch,h,w,c = X[0].shape
		# print("X.shape",h,w,c)
		assert len(X) == self.num_inputs
		result = []
		encoded_reps,rec = self.segnets[0].call(X[0])
		encoded_reps = tf.expand_dims(encoded_reps,1)
		if classification == False:
			result.append(rec)
		for i in range(self.num_inputs-1):
			enc,rec = self.segnets[i+1].call(X[i+1])
			enc = tf.expand_dims(enc,1)
			encoded_reps = tf.concat([encoded_reps,enc],axis=1)
			if classification == False:
				result.append(rec)  #Appending the reconstructed result to return 
		if classification == False:
			result.append(tf.reshape(self.reconstruct_image(encoded_reps),(batch,h,w,c)))   #Appending final image
		#Uncomment the two lines below to enable classification
		else:
			result.append(self.predict_label(encoded_reps))     #Appending final labels
		return result

	def loss_reconstruction(self,X,Y,beta=0.0):
		# print(X.shape,Y.shape)
		#Pixel-wise l2 loss
		# return  tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum((X-Y)**2,
			# axis=-1),axis=-1),axis=-1,keepdims=True)    #see if keepdims is required
		return (1-beta)*tf.math.reduce_sum((X-Y)**2)/(X.shape[1]*X.shape[2]*X.shape[3]) + beta*tf.math.reduce_sum(tf.math.abs(X-Y))/(X.shape[1]*X.shape[2]*X.shape[3])

	def loss_classification(self,X,labels):
		return (-1*tf.reduce_mean(labels*(tf.math.log(X+1e-5)) + (1-labels)*(tf.math.log(1-X+1e-5))))

	def train_on_batch(self,X,labels,optimizer,classification=False):
		# If classification is True, labels is a one-hot of the class
		# Else it is the target image
		with tf.GradientTape() as tape:
			
			result = self.call(X,classification)
			losses = []
			loss = 0
			if classification == False:
				loss = self.loss_reconstruction(X[0],result[0])
				losses.append(loss)
			for i in range(self.num_inputs-1):
				loss += self.loss_reconstruction(X[i+1],result[i+1])
				losses.append(self.loss_reconstruction(X[i+1],result[i+1]))
			loss += self.loss_reconstruction(result[self.num_inputs],labels)
			losses.append(self.loss_reconstruction(result[self.num_inputs],labels))

			else:
				loss += self.loss_classification(result[-1],labels)
				losses.append(self.loss_classification(result[-1],labels))

		if classification == False:
			grads = tape.gradient(loss,self.trainableVariables)
			grads_and_vars = zip(grads, self.trainableVariables)
		else:
			grads = tape.gradient(loss,self.predict_label.trainable_variables)
			grads_and_vars = zip(grads, self.predict_label.trainable_variables)

		optimizer.apply_gradients(grads_and_vars)

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
		return (tf.math.argmax(result[-1],axis=1).numpy()==np.argmax(Y_labels,axis=1)).sum(),losses
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
		for i in range(self.segnets):
			self.segnets[i].save("{}/Segnet-{}".format(modelDir,i))
		pickle.dump(self.reconstruct_image.get_weights(),
			open("{}/Reconstruction-Model".format(modelDir),"wb"))
		pickle.dump(self.predict_label.get_weights(),
			open("{}/Prediction-Model".format(modelDir),"wb"))


	def load_model(self,modelDir):
		for i in range(self.segnets):
			self.segnets[i].load_model("{}/Segnet-{}".format(modelDir,i))
		rec_train_vars = pickle.load(open("{}/Reconstruction-Model".format(modelDir),"rb"))
		pred_train_vars = pickle.load(open("{}/Prediction-Model".format(modelDir),"rb"))
		for l in self.reconstruct_image.layers:
			# weights = l.get_weights()
			weights = rec_train_vars
			l.set_weights(weights)
		for l in self.predict_label.layers:
			# weights = l.get_weights()
			weights = pred_train_vars
			l.set_weights(weights)
		self.TrainableVarsSet = False
