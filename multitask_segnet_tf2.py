from  tf.keras import Sequential
from tf.keras.layers import Conv2D,ReLU,BatchNormalization
from layers import *
# import argparse 

# parser = argparse.ArgumentParser(description='Multi-task: Split')
# parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
# parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
# parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
# parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
# opt = parser.parse_args()

class SegNet(Sequential):
	def __init__(self,nettype='standard'):
        super(SegNet, self).__init__()
        if nettype == 'wide':
            filter = [64, 128, 256, 512, 1024]
        else:
            filter = [64, 128, 256, 512, 512]


        self.max_indices = []	#TODO can we refactor to make this a tensor?

        # defining the convolution part of encoder-decoder blocks
        #TODO Refactor for better model making
        self.conv_block_enc = tf.keras.Sequential([self.conv_layer(filter[0]),self.conv_layer(filter[0])])
        self.conv_block_dec = tf.keras.Sequential([self.conv_layer(filter[-1]),
        										  self.conv_layer(filter[-1]),
        										  self.conv_layer(filter[-1])])	
        for i in range(4):	#TODO Refactor for better model making
        	j = 4 - i
            if i == 0:
                self.conv_block_enc.add(Sequential([self.conv_layer(filter[i + 1]),
                								    self.conv_layer(filter[i + 1])]))
                self.conv_block_dec.add(Sequential([self.conv_layer(filter[j]),
                									self.conv_layer(filter[j]),
                									self.conv_layer(filter[j-1])]))
            elif j==1:
                self.conv_block_enc.add(Sequential([self.conv_layer(filter[i + 1]),
                									self.conv_layer(filter[i + 1]),
                                                    self.conv_layer(filter[i + 1])]))
                self.conv_block_dec.add(Sequential([self.conv_layer(filter[j]),
                                                    self.conv_layer(filter[j-1])]))
            elif j==0:
                self.conv_block_enc.add(Sequential([self.conv_layer(filter[i + 1]),
                									self.conv_layer(filter[i + 1]),
                                                    self.conv_layer(filter[i + 1])]))
                self.conv_block_dec.add(Sequential([self.conv_layer(filter[j])]))           	
            else:
                self.conv_block_enc.add(Sequential([self.conv_layer(filter[i + 1]),
                									self.conv_layer(filter[i + 1]),
                                                    self.conv_layer(filter[i + 1])]))
                self.conv_block_dec.add(Sequential([self.conv_layer(filter[j]),
                                                    self.conv_layer(filter[j]),
                                                    self.conv_layer(filter[j-1])]))

        #Uncomment for task specific layers  (Won't be using here)  
        # self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=self.class_nb, kernel_size=1, padding=0))
        # self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        # self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
        #                                 nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))

        # define pooling and unpooling functions
        self.down_sampling = MaxPool2D(ksize=(2,2), strides=(2,2),padding='same')
        self.up_sampling = MaxUnpool2D(size=(2,2))

 
    def conv_layer(self, channel):
        if opt.type == 'deep':
            conv_block = tf.keras.Sequential(
                [Conv2D(filters=channel, kernel_size=3, padding="valid"),
                BatchNormalization(axis=-1),
                ReLU(),
                Conv2D(filters=channel, kernel_size=3, padding="valid"),
                BatchNormalization(axis=-1),	#SANITY CHECK, see that channels are LAST!!!
                ReLU()]
            )
        else:
            conv_block = tf.keras.Sequential(
                [Conv2D(filters=channel, kernel_size=3, padding="valid"),
                BatchNormalization(axis=-1),
                ReLU()]
            )
        return conv_block

    def call(self,X):
    	# Returns 2 tensors, one is the encoded representation, other 
    	# is the reconstruction
    	for layer in self.conv_block_enc.layers:
    		X = layer.call(X)
    		X,ind = self.down_sampling(X)
    		self.max_indices.append(ind)
    	X_enc = X 	#POTENTIAL BUG IF SHALLOW COPY
    	self.max_indices = self.max_indices[::-1]	#Reverse list for easier access
    	for idx,layer in enumerate(self.conv_block_enc.layers):
    		X = layer.call(X)
    		X = self.down_sampling(X,self.max_indices[idx])
    	return [X_enc,X]

    def trainable_variables(self):
    	return self.conv_block_enc.trainable_variables + self.conv_block_dec.trainable_variables
