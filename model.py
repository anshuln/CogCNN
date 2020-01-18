#TODO Rename file
import tensorflow as tf
from multitask_segnet_tf2 import *
from layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Reshape,Dropout,BatchNormalization,Conv2D,MaxPool2D,LeakyReLU,Conv2DTranspose
import numpy as np
import pickle
from layers import ReshapeAndConcat

class MultiTaskModel(Sequential):
	def __init__(self,image_shape,num_labels,num_inputs=4,trainableVariables=None,attention=None,two_stage=False,pix2pix=False):
		#num_inputs refers to input channels(edge,texture etc.)
		#image_shape is the shape of 1 image for reconstruction
		#TODO - kwargs support for segnet initializations
		super(MultiTaskModel, self).__init__()
		self.num_inputs = num_inputs    
		self.image_shape = image_shape
		self.segnets = []
		self.attention = attention
		self.pix2pix = pix2pix
		self.two_stage = two_stage

		if self.attention == "self":      
			self.attention_gates_rec = []
			self.attention_gates_pred = []
		if trainableVariables is None:
			self.trainableVariables = []    #Not to be confused with trainable_variables, which is read-only
		else:
			self.trainableVariables = trainableVariables
		for i in range(num_inputs):
			#TODO make better attention layers.
			self.segnets.append(SegNet())
			if self.attention == "self":
				self.attention_gates_rec.append(SelfAttention([Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),Conv2D(filters=512, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid")]))
				self.attention_gates_pred.append(SelfAttention([Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),Conv2D(filters=512, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid")]))
			elif self.attention == "multi":
				self.attention_gates_rec = CompleteAttention(layers=[Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="relu"),Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),Conv2D(filters=512*self.num_inputs, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid")],num_streams=self.num_inputs)
				self.attention_gates_pred = CompleteAttention(layers=[Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="relu"),Conv2D(filters=128, kernel_size=3, padding="same",kernel_initializer='glorot_normal'),Conv2D(filters=512*self.num_inputs, kernel_size=3, padding="same",kernel_initializer='glorot_normal',activation="sigmoid")],num_streams=self.num_inputs)
		print("Image_Shape",image_shape)
		#TODO fix the reconstruction net to have some conv layers
		self.reconstruct_image = Sequential([#Conv2D(filters=512, kernel_size=2,padding="same",activation="relu"),Conv2D(filters=256, kernel_size=2,padding="valid"),
				Flatten(),BatchNormalization(axis=-1),Dense(1024),Dense(image_shape[0]*image_shape[1]*image_shape[2],activation='sigmoid')])
		#Change activation to relu
		#Uncomment the two lines below to enable classification
		self.predict_label = Sequential([Conv2D(filters=128, kernel_size=2,padding="same",activation="relu"),Conv2D(filters=128, kernel_size=2,padding="valid"),Flatten(),#Dense(1000),BatchNormalization(axis=-1),
				Dense(num_labels,activation='softmax')])    #The loss function uses softmax, final preds as well
		if self.pix2pix:
			self.discriminator = Sequential()
			disc_layers = [ReshapeAndConcat(),Conv2D(128,3,padding="valid"),MaxPool2D(pool_size=(2,2)),LeakyReLU(),BatchNormalization(axis=-1),Conv2D(128,3,padding="valid"),MaxPool2D(pool_size=(2,2)),LeakyReLU(),Flatten(),BatchNormalization(axis=-1),Dense(32,activation='relu'),BatchNormalization(axis=-1),Dense(1,activation='sigmoid',kernel_initializer="glorot_normal")]
			for l in disc_layers:
				self.discriminator.add(l)

	def examine_disc(self,X):
		for l in self.discriminator.layers[:-1]:
			X = l.call(X)
		return (X)

	def regularizer_naive(self,means,meansum,indx):
		return tf.math.abs(means[indx]/meansum)
	def regularizer_ratio(self,meansr,meansp,meansumr,meansump):
		sum = tf.math.abs(meansr[0]/meansumr-meansp[0]/meansump)
		for i in range(1,self.num_inputs):
			sum += tf.math.abs(meansr[i]/meansumr-meansp[i]/meansump)
		return sum
	def setTrainableVariables(self,trainableVariables=None):
		if trainableVariables is not None:
			self.trainableVariables = trainableVariables
			return
		for i in range(self.num_inputs):    
			print("On segnet",i)
			self.trainableVariables += self.segnets[i].trainable_variables
		if self.attention == "self":
			for i in range(self.num_inputs):    
				self.trainableVariables += self.attention_gates_rec[i].trainable_variables
				self.trainableVariables += self.attention_gates_pred[i].trainable_variables
		elif self.attention == "multi":
			self.trainableVariables += self.attention_gates_rec.trainable_variables
			self.trainableVariables += self.attention_gates_pred.trainable_variables

		self.trainableVariables += self.reconstruct_image.trainable_variables
		self.trainableVariables += self.predict_label.trainable_variables 

		if self.pix2pix:
			self.disc_train_vars = []
			for l in self.discriminator.layers:
				self.disc_train_vars+=l.trainable_variables
	
	# @tf.function
	def build(self,X):
		batch,h,w,c = X[0].shape
		assert len(X) == self.num_inputs
		result = []
		encoded_reps,rec = self.segnets[0].call(X[0])
		if self.attention=="self":
			encoded_reps_rec = self.attention_gates_rec[0].call(encoded_reps)
			encoded_reps_pred = self.attention_gates_pred[0].call(encoded_reps)
			# encoded_reps_rec = tf.expand_dims(encoded_reps_rec,1)
			# encoded_reps_pred = tf.expand_dims(encoded_reps_pred,1)
		else:
			# encoded_reps = tf.expand_dims(encoded_reps,1) 
			pass
		result.append(rec)
		for i in range(self.num_inputs-1):
			enc,rec = self.segnets[i+1].call(X[i+1])
			if self.attention=="self":
				encoded_attended_rec = self.attention_gates_rec[i+1].call(encoded_reps)
				encoded_attended_pred = self.attention_gates_pred[i+1].call(encoded_reps)
				# encoded_attended_rec = tf.expand_dims(encoded_attended_rec,1)
				# encoded_attended_pred = tf.expand_dims(encoded_attended_pred,1)
				print("-----------\n",encoded_reps_rec.shape,encoded_attended_rec.shape)
				encoded_reps_rec = tf.concat([encoded_reps_rec,encoded_attended_rec],axis=-1)
				encoded_reps_pred = tf.concat([encoded_reps_pred,encoded_attended_pred],axis=-1)
			else:
				# enc = tf.expand_dims(enc,1)
				encoded_reps = tf.concat([encoded_reps,enc],axis=-1)

			result.append(rec)
		if self.attention == "multi":
			encoded_reps_rec,_,_ = self.attention_gates_rec(encoded_reps)
			encoded_reps_pred,_,_ = self.attention_gates_pred(encoded_reps)

		if self.attention is not None:
			result.append(tf.reshape(self.reconstruct_image(encoded_reps_rec),(batch,h,w,c)))   #               
			result.append(self.predict_label(encoded_reps_pred))     #Appending final labels
			if self.pix2pix:
				result.append(encoded_reps_rec) #Needed for pix2pix
		else:
			result.append(tf.reshape(self.reconstruct_image(encoded_reps),(batch,h,w,c)))   #
			result.append(self.predict_label(encoded_reps))     #Appending final labels
			if self.pix2pix:
				result.append(encoded_reps) #Needed for pix2pix
		if self.pix2pix:
			self.discriminator.call((result[-1],result[self.num_inputs]))
			log = open("log_pix2pix.txt","w")
			# log.write("Rec {}\n".format(self.discriminator((result[-1],result[self.num_inputs]))))
			# log.write("weights {}\n".format(self.discriminator.layers[-1].trainable_variables))
			log.write("\n")
			log.close()
	@tf.function
	def call(self,X,classification=False):
		#X is a LIST of the dimension [batch*h*w*c]*num_inputs
		#TODO check if this gives us correct appending upon flatten
		#TODO refactor to make everything a tensor
		batch,h,w,c = X[0].shape
		assert len(X) == self.num_inputs
		result = []
		encoded_reps,rec = self.segnets[0].call(X[0])
		if self.attention=="self":
			encoded_reps_rec = self.attention_gates_rec[0].call(encoded_reps)
			encoded_reps_pred = self.attention_gates_pred[0].call(encoded_reps)
			# encoded_reps_rec = tf.expand_dims(encoded_reps_rec,1)
			# encoded_reps_pred = tf.expand_dims(encoded_reps_pred,1)
		else:
			# encoded_reps = tf.expand_dims(encoded_reps,1) 
			pass
		result.append(rec)
		for i in range(self.num_inputs-1):
			enc,rec = self.segnets[i+1].call(X[i+1])
			if self.attention=="self":
				encoded_attended_rec = self.attention_gates_rec[i+1].call(encoded_reps)
				encoded_attended_pred = self.attention_gates_pred[i+1].call(encoded_reps)
				# encoded_attended_rec = tf.expand_dims(encoded_attended_rec,1)
				# encoded_attended_pred = tf.expand_dims(encoded_attended_pred,1)
				encoded_reps_rec = tf.concat([encoded_reps_rec,encoded_attended_rec],axis=-1)
				encoded_reps_pred = tf.concat([encoded_reps_pred,encoded_attended_pred],axis=-1)
			else:
				# enc = tf.expand_dims(enc,1)
				encoded_reps = tf.concat([encoded_reps,enc],axis=-1)

			result.append(rec)
		if self.attention == "multi":
			encoded_reps_rec,meansr,meansumr = self.attention_gates_rec(encoded_reps)
			encoded_reps_pred,meansp,meansump = self.attention_gates_pred(encoded_reps)

		if self.attention is not None:
			result.append(tf.reshape(self.reconstruct_image(encoded_reps_rec),(batch,h,w,c)))   #               
			result.append(self.predict_label(encoded_reps_pred))     #Appending final labels
			if self.pix2pix:
				result.append(encoded_reps_rec) #Needed for pix2pix
			if self.attention == "multi":
				result.append(meansr)
				result.append(meansumr)
				result.append(meansp)
				result.append(meansump)
		else:
			result.append(tf.reshape(self.reconstruct_image(encoded_reps),(batch,h,w,c)))   #
			result.append(self.predict_label(encoded_reps))     #Appending final labels
			if self.pix2pix:
				result.append(encoded_reps) #Needed for pix2pix

		return result

	def loss_reconstruction(self,X,Y,beta=0.0):
		# print(X.shape,Y.shape)
		#Pixel-wise l2 loss
		# return  tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum((X-Y)**2,
			# axis=-1),axis=-1),axis=-1,keepdims=True)    #see if keepdims is required
		return (1-beta)*tf.math.reduce_sum((X-Y)**2)/(X.shape[1]*X.shape[2]*X.shape[3]) + beta*tf.math.reduce_sum(tf.math.abs(X-Y))/(X.shape[1]*X.shape[2]*X.shape[3])

	def loss_classification(self,X,labels):
		return (-1*tf.reduce_mean(labels*(tf.math.log(X+1e-10)) + (1-labels)*(tf.math.log(1-X+1e-10))))

	def generator_loss(self,disc_generated_output, gen_output, target,LAMBDA=0.1):
		gan_loss = self.loss_classification(disc_generated_output,tf.ones_like(disc_generated_output))

		# mean absolute error
		l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

		total_gen_loss = gan_loss + (LAMBDA * l1_loss)

		return total_gen_loss
	def discriminator_loss(self,disc_real_output, disc_generated_output):
		real_loss = self.loss_classification(disc_real_output,tf.ones_like(disc_real_output))

		generated_loss = self.loss_classification(disc_generated_output,tf.zeros_like(disc_generated_output))
		# log = open("log_pix2pix.txt","a")
		# log.write("Rec {}\n".format(self.examine_disc((result[-1],result[self.num_inputs]))))
		# log.write("Actual {}\n".format(self.examine_disc((result[-1],Y_image))))  
		# log.write("Rec {}\n".format(self.discriminator((result[-1],result[self.num_inputs]))))
		# log.write("Actual {}\n".format(self.discriminator((result[-1],Y_image))))
		# log.write("loss {}\n".format(generated_loss))
		# log.write("weights {}\n".format(self.discriminator.layers[-1].trainable_variables))
		# log.close()
		

		total_disc_loss = real_loss + generated_loss

		return total_disc_loss

	#TODO make this tf.function
	def train_on_batch(self,X,Y_image,Y_labels,optimizer,classification=False):
		# Y needs to be a list of [img,labels]
		with tf.GradientTape(persistent=True) as tape:
			
			result = self.call(X)
			losses = []
			loss = 0
			loss_disc = 0
			if self.two_stage:
				if classification==False:
					loss = self.loss_reconstruction(X[0],result[0])
					losses.append(loss)
					for i in range(self.num_inputs-1):
						loss += self.loss_reconstruction(X[i+1],result[i+1])
						losses.append(self.loss_reconstruction(X[i+1],result[i+1]))
					#TODO FIX THIS, since result[-1] is not the required thing anymore
					if self.pix2pix:
						disc_real_output = self.discriminator.call((result[-1],Y_image))
						disc_generated_output = self.discriminator.call((result[-1],result[self.num_inputs]))
						loss += self.generator_loss(disc_generated_output,result[self.num_inputs],Y_image)
					else:
						loss += self.loss_reconstruction(result[self.num_inputs],Y_image)
						losses.append(self.loss_reconstruction(result[self.num_inputs],Y_image))
				else:
					#Uncomment the two lines below to enable classification
					loss += self.loss_classification(result[self.num_inputs+1],Y_labels)
					losses.append(self.loss_classification(result[self.num_inputs+1],Y_labels))
					if self.attention == 'multi':
						loss += self.regularizer_ratio(result[-2],result[-4],result[-1],result[-3]) #result[-1],result[-3]) #TODO Have this tunable

			else:
				loss = self.loss_reconstruction(X[0],result[0])
				losses.append(loss)
				for i in range(self.num_inputs-1):
					loss += self.loss_reconstruction(X[i+1],result[i+1])
					losses.append(self.loss_reconstruction(X[i+1],result[i+1]))
				if self.pix2pix:
					disc_real_output = self.discriminator.call((result[-1],Y_image))
					disc_generated_output = self.discriminator.call((result[-1],result[self.num_inputs]))
					loss += self.generator_loss(disc_generated_output,result[self.num_inputs],Y_image)
					loss_disc += self.discriminator_loss(disc_real_output,disc_generated_output)
					losses.append(loss_disc)
					losses.append(self.generator_loss(disc_generated_output,result[self.num_inputs],Y_image))
				else:
					loss += self.loss_reconstruction(result[self.num_inputs],Y_image)
					losses.append(self.loss_reconstruction(result[self.num_inputs],Y_image))    
				#Uncomment the two lines below to enable classification

				loss += self.loss_classification(result[self.num_inputs+1],Y_labels)
				losses.append(self.loss_classification(result[self.num_inputs+1],Y_labels))
				if self.attention == 'multi':
					loss += self.regularizer_ratio(result[-2],result[-4],result[-1],result[-3]) #TODO Have this tunable

				# loss += self.regularizer_ratio(result[-4],result[-3],indx=0)

		if self.two_stage==True and classification==True:
			train_vars = self.predict_label.trainable_variables
			if self.attention=="self":
				for att in self.attention_gates_pred:
					train_vars += att.trainable_variables
			elif self.attention == "multi":
				train_vars += self.attention_gates_pred.trainable_variables

			grads = tape.gradient(loss,train_vars)
			grads_and_vars = zip(grads, train_vars)

			optimizer.apply_gradients(grads_and_vars)


		else:
			grads = tape.gradient(loss,self.trainableVariables)
			grads_and_vars = zip(grads, self.trainableVariables)
			optimizer.apply_gradients(grads_and_vars)

		if self.pix2pix:
			grad_disc = tape.gradient(loss_disc,self.disc_train_vars)
			grads_and_vars_disc = zip(grad_disc, self.disc_train_vars)
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
		if self.pix2pix:
			log = open("log_pix2pix.txt","a")
			log.write("Rec {}\n".format(self.examine_disc((result[-1],result[self.num_inputs]))))
			log.write("Actual {}\n".format(self.examine_disc((result[-1],Y_image))))    
			log.write("Rec {}\n".format(self.discriminator((result[-1],result[self.num_inputs]))))
			log.write("Actual {}\n".format(self.discriminator((result[-1],Y_image))))
			log.write("weights {}\n".format(self.discriminator.layers[-1].trainable_variables))
			log.close()
			return (tf.math.argmax(result[-2],axis=1).numpy()==np.argmax(Y_labels,axis=1)).sum(),losses
		else:
			return (tf.math.argmax(result[-1],axis=1).numpy()==np.argmax(Y_labels,axis=1)).sum(),losses
			
		# return losses
	def getAttentionMap(self,X):
		# Saves attention map for X
		if self.attention == "self":
			attention_maps_rec = []
			attention_maps_pred = []
			batch,h,w,c = X[0].shape
			# print("X.shape",h,w,c)
			assert len(X) == self.num_inputs
			result = []
			encoded_reps,rec = self.segnets[0].call(X[0])
			attention = self.attention_gates_rec[0].get_attention_map(encoded_reps).numpy()
			attention_maps_rec.append(attention)
			for i in range(self.num_inputs-1):
				enc,rec = self.segnets[i+1].call(X[i+1])
				attention = self.attention_gates_rec[i+1].get_attention_map(encoded_reps).numpy()
				attention_maps_rec.append(attention)  #Appending the reconstructed result to return 

			encoded_reps,rec = self.segnets[0].call(X[0])
			attention = self.attention_gates_pred[0].get_attention_map(encoded_reps).numpy()
			attention_maps_pred.append(attention)
			for i in range(self.num_inputs-1):
				enc,rec = self.segnets[i+1].call(X[i+1])
				attention = self.attention_gates_pred[i+1].get_attention_map(encoded_reps).numpy()
				attention_maps_pred.append(attention)  #Appending the reconstructed result to return 
		else:
			encoded_reps,_ = self.segnets[0].call(X[0]) 
			for i in range(self.num_inputs-1):
				enc,_      = self.segnets[i+1].call(X[i+1])
				encoded_reps = tf.concat([encoded_reps,enc],axis=-1)
			
			attention_maps_rec  = self.attention_gates_rec(encoded_reps)[0].numpy()
			attention_maps_pred = self.attention_gates_pred(encoded_reps)[0].numpy()
		result = tf.math.argmax(self.predict_label(tf.concat(attention_maps_pred,axis=-1)),axis=1)

		return np.array(attention_maps_rec),np.array(attention_maps_pred),result.numpy()

	def save(self,modelDir):
		for i in range(len(self.segnets)):
			self.segnets[i].save("{}/Segnet-{}".format(modelDir,i))
		if self.attention == 'self':
			for i in range(len(self.segnets)):
				pickle.dump(self.attention_gates_pred[i].get_weights(),
					open("{}/Attention-Pred-{}".format(modelDir,i),"wb"))
				pickle.dump(self.attention_gates_rec[i].get_weights(),
					open("{}/Attention-Rec-{}".format(modelDir,i),"wb"))
		elif self.attention == 'multi':
			pickle.dump(self.attention_gates_pred.get_weights(),
				open("{}/Attention-Pred".format(modelDir,i),"wb"))
			pickle.dump(self.attention_gates_rec.get_weights(),
				open("{}/Attention-Rec".format(modelDir,i),"wb"))

		pickle.dump(self.reconstruct_image.get_weights(),
			open("{}/Reconstruction-Model".format(modelDir),"wb"))
		pickle.dump(self.predict_label.get_weights(),
			open("{}/Prediction-Model".format(modelDir),"wb"))


	def load_model(self,modelDir,attention,two_stage,pix2pix):
		for i in range(len(self.segnets)):
			self.segnets[i].load_model("{}/Segnet-{}".format(modelDir,i))
		rec_train_vars = pickle.load(open("{}/Reconstruction-Model".format(modelDir),"rb"))
		pred_train_vars = pickle.load(open("{}/Prediction-Model".format(modelDir),"rb"))
		# for l in self.reconstruct_image.layers:
		#   weights = rec_train_vars
		self.reconstruct_image.set_weights(rec_train_vars)
		# for l in self.predict_label.layers:
		#   weights = pred_train_vars
		self.predict_label.set_weights(pred_train_vars)
		self.TrainableVarsSet = False
		self.pix2pix   = pix2pix
		self.attention = attention
		self.two_stage = two_stage

		if self.attention == "multi":
			pred_gates = pickle.load(open("{}/Attention-Pred".format(modelDir),"rb"))
			rec_gates = pickle.load(open("{}/Attention-Rec".format(modelDir),"rb"))
			self.attention_gates_pred.set_weights(pred_gates)
			self.attention_gates_rec.set_weights(rec_gates)
		elif self.attention == 'self':
			raise NotImplementedError

