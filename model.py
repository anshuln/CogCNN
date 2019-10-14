#TODO Rename file
import tensorflow as tf
from multitask_segnet_tf2 import *
from tf.keras import Sequential
from tf.keras.layers import Flatten,Dense

class MultiTaskModel(Sequential):
	def __init__(self,num_inputs=4,image_shape,num_labels):
		#num_inputs refers to input channels(edge,texture etc.)
		#image_shape is the shape of 1 image for reconstruction
		#TODO - kwargs support for segnet initializations
		super(MultiTaskModel, self).__init__()
		self.num_inputs = num_inputs	
		self.image_shape = image_shape
		self.segnets = []
		self.trainable_variables = []
		for i in range(num_inputs):
			self.segnets.append(SegNet())
			self.trainable_variables += self.segnets[-1].trainable_variables
		self.reconstruct_image = Sequential([Flatten(),Dense(1000),
				Dense(image_shape[0]*image_shape[1],image_shape[2])])
		self.trainable_variables += self.reconstruct_image.trainable_variables
		self.predict_label = Sequential([Flatten(),Dense(1000),
				Dense(num_labels,activation='softmax')])
		self.trainable_variables += self.predict_label.trainable_variables 

	def call(self,X):
		#X is a LIST of the dimension [batch*h*w*c]*num_inputs
		#TODO check if this gives us correct appending upon flatten
		#TODO refactor to make everything a tensor
		batch,h,w,c = X[0].shape
		assert len(X) == self.num_inputs
		result = []
		encoded_reps,rec = self.segnets[0].call(X[0])
		encoded_reps = tf.expand_dims(encoded_reps,1)
		reconstruction = tf.expand_dims(rec,1)
		result.append(rec)
		for i in range(self.num_inputs-1):
			enc,rec = self.segnets[i+1].call(X[i+1])
			encoded_reps = tf.concat([encoded_reps,enc],axis=1)
			result.append(rec)
			reconstruction = tf.concat([reconstruction,rec],axis=1)

		result.append(tf.reshape(self.reconstruct_image(reconstruction),(batch,h,w,c)))
		result.append(self.predict_label(encoded_reps))

	def loss_reconstruction(self,X,Y):
		#Pixel-wise l2 loss
		return  tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_sum((X-Y)**2,
			axis=-1),axis=-1),axis=-1,keepdims=True)	#see if keepdims is required

	def loss_classification(self,X,labels):
		pass

	def train_on_batch(self,X,Y,optimizer):
		# Y needs to be a list of [img,labels]
		with tf.GradientTape() as tape:
			result = self.call(X)
			loss = self.loss_reconstruction(X[0],result[0])
			for i in range(self.num_inputs-1):
				loss += self.loss_reconstruction(X[i+1],result[i+1])
			loss += self.loss_reconstruction(result[self.num_inputs],Y[0])
			loss += self.loss_classification(result[self.num_inputs+1],Y[1])
		grads = tape.gradient(loss,self.trainable_variables)
		grads_and_vars = zip(grads, self.trainable_variables)
		optimizer.apply_gradients(grads_and_vars)
		return loss

    # def fit(self, X,Y, batch_size=32,epochs=1,verbose=1,validation_split=0.0,
    # validation_data=None,
    # shuffle=True,
    # initial_epoch=0,
    # steps_per_epoch=None,
    # validation_steps=None,
    # validation_freq=1,
    # optimizer=tf.optimizers.Adam(),**kwargs): 
    #     '''
    #     Fits the model on dataset `X (not a generator)
    #     Note - for very big datasets, the function will give OOM, 
    #            consider using a generator
    #     Args-
    #     X - Data to be fitted. Maybe one of the following-
    #             tf.EagerTensor
    #             np.ndarray
    #     batch_size - Number of elements in each minibatch
    #     verbose - Logging level
    #     validation_split - Amount of data to be used for validation in each epoch
    #                        For tensors or arrays, data is extracted from initial part of dataset.
    #     shuffle - Should training data be shuffled before mini-batches are extracted
    #     steps_per_epoch - Number of training steps per epoch. Used mainly for generators.
    #     validation_steps - Number of validation steps per epoch. Used mainly for generators.

    #     '''
    #     # TODO add all callbacks from tf.keras.Model.fit 
    #     # TODO return a history object instead of array of losses
    #     all_losses = []
    #     if validation_split > 0 and validation_data is None:
    #         validation_data = X[:int(len(X)*validation_split)]
    #         X = X[int(len(X)*validation_split):]

    #     epoch_gen = range(initial_epoch,epochs)
    #     if verbose == 1:
    #         epoch_gen = tqdm(epoch_gen)
    #     batch_size = min(batch_size,X.shape[0]) #Sanity check
    #     num_batches = X.shape[0] // batch_size
    #     if steps_per_epoch == None:
    #         steps_per_epoch = num_batches
    #     val_count = 0

    #     for j in epoch_gen:
    #         if shuffle == True:
    #             X = np.random.permutation(X)    #Works for np.ndarray and tf.EagerTensor, however, turns everything to numpy
    #         #Minibatch gradient descent
    #         range_gen = range(steps_per_epoch)
    #         if verbose == 2:
    #             range_gen = tqdm(range_gen)
    #         for i in range_gen:    
    #             losses = []
    #             loss = self.train_on_batch(X[i*batch_size:(i+1)*(batch_size)],optimizer)
    #             losses.append(loss.numpy())
    #         loss = np.mean(losses)  
    #         all_losses+=losses
    #         to_print = 'Epoch: {}/{}, training_loss: {}'.format(j,epochs,loss)
    #         if validation_data is not None and val_count%validation_freq==0:
    #             val_loss = self.loss(validation_data)
    #             to_print += ', val_loss: {}'.format(val_loss.numpy())   #TODO return val_loss somehow
    #         if verbose == 2:
    #             print(to_print)
    #         val_count+=1
    #     return all_losses

    # def fit_generator(self, generator,steps_per_epoch=None,initial_epoch=0,
    #     epochs=1,
    #     verbose=1,validation_data=None,
    #     validation_freq=1,
    #     shuffle=True,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False,
    #     optimizer=tf.optimizers.Adam(),
    #     **kwargs): 
    #     '''
    #     Fits model on the data generator `generator
    #     IMPORTANT - Please consider using invtf.data.load_image_dataset()
    #     Args - 
    #     generator - tf.data.Dataset, tf.keras.utils.Sequence or python generator
    #     validation_data - same type as generator
    #     steps_per_epoch - int, number of batches per epoch.
    #     '''
    #     #TODO add callbacks and history
    #     all_losses = []
    #     if isinstance(generator,tf.keras.utils.Sequence):
    #         enqueuer = tf.keras.utils.OrderedEnqueuer(generator,use_multiprocessing,shuffle)    
    #         if steps_per_epoch == None:
    #             steps_per_epoch = len(generator)    #TODO test this, see if it works for both Sequence and Dataset
    #         enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    #         output_generator = enqueuer.get()               
    #     elif isinstance(generator,tf.data.Dataset):
    #         output_generator = iter(generator)
    #     else:
    #         enqueuer = tf.keras.utils.GeneratorEnqueuer(generator,use_multiprocessing)  # Can't shuffle here!
    #         enqueuer.start(workers=workers, max_queue_size=max_queue_size)  
    #         output_generator = enqueuer.get()   
    #     if validation_data is not None:     #Assumption that validation data and generator are same type
    #         if isinstance(generator,tf.keras.utils.Sequence):
    #             val_enqueuer = tf.keras.utils.OrderedEnqueuer(validation_data,use_multiprocessing,shuffle)  
    #             val_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    #             val_generator = val_enqueuer.get()              
    #         elif isinstance(generator,tf.data.Dataset):
    #             val_generator = iter(val_generator)
    #         else:
    #             val_enqueuer = tf.keras.utils.GeneratorEnqueuer(validation_data,use_multiprocessing)    # Can't shuffle here!
    #             val_enqueuer.start(workers=workers, max_queue_size=max_queue_size)  
    #             val_generator = val_enqueuer.get()  

    #     if steps_per_epoch == None:
    #         raise ValueError("steps_per_epoch cannot be None with provided generator")
    #     epoch_gen = range(initial_epoch,epochs)
    #     if verbose == 1:
    #         epoch_gen = tqdm(epoch_gen)
    #     for j in epoch_gen:
    #         range_gen = range(steps_per_epoch)
    #         if verbose == 2:
    #             range_gen = tqdm(range_gen)
    #         for i in range_gen:
    #             losses = []
    #             loss = self.train_on_batch(next(output_generator),optimizer)
    #             losses.append(loss.numpy())
    #         loss = np.mean(losses)  
    #         to_print = 'Epoch: {}/{}, training_loss: {}'.format(j,epochs,loss)
    #         if validation_data is not None and val_count%validation_freq==0:
    #             val_loss = self.loss(next(val_generator))
    #             to_print += ', val_loss: {}'.format(val_loss.numpy())   #TODO return val_loss somehow
    #         if verbose == 2:
    #             print(to_print)
    #         all_losses+=losses
    #         val_count+=1
    #     try:
    #         if enqueuer is not None:
    #             enqueuer.stop()         
    #     except:
    #         pass
    #     return all_losses









