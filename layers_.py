import torch
from torch import nn

class Flatten(nn.module):
	def forward(self, x):
		x = x.view(x.size()[0], -1)
		return x

class CompleteAttention(nn.module):
	def __init__(self,layers,num_streams):
		super(CompleteAttention, self).__init__()
		self.model = nn.Sequential(layers)
		self.num_streams = num_streams

	def forward(self,x):
		attention_map,means,meansum = self.get_attention_map(X)

		return X*attention_map,means,meansum

	def get_attention_map(self,x):
		attention_map = self.model(X)	
		b,c,h,w 	  = list(attention_map.size())
		means 		  = []
		for i in range(self.num_streams):
			means.append(torch.mean(attention_map[:,i*(c//self.num_streams):(i+1)*(c//self.num_streams),:,:]))
		
		meansum = 0
		for i in range(self.num_streams):
			meansum += means[i]

		return attention_map,means,meansum

	def set_train_false(self):
		for p in self.model.parameters():
			p.requires_grads = False
