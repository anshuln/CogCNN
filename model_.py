import torch 
from torch import nn 

class MultiTaskModel(nn.module):
	def __init__(self,image_shape,num_labels,num_inputs=4):
		super(MultiTaskModel,self).__init__()
		self.image_shape = image_shape 
		self.num_inputs = num_inputs
		self.num_labels = num_labels

		self.segnets = []

		for i in range(num_inputs):
			self.segnets.append(SegNet())

		self.attention_gates_rec  = CompleteAttention([Conv2d(512,128,3,padding=),Conv2d(128,512,3,padding=)]) 
		self.attention_gates_pred = CompleteAttention([Conv2d(512,128,3,padding=),Conv2d(128,512,3,padding=)])

		self.reconstruct_image = nn.Sequential(Flatten(),BatchNorm1d(512*),Linear(512*,1024),Linear(1024,image_shape[0]*image_shape[1]*image_shape[2]))	#Add sigmoid

		self.predict_label = nn.Sequential(Conv2d(512,128,2,padding=),Conv2d(128,128,2),Flatten(),Linear(,num_labels
			))
	def forward(self,x):

		batch,num_ip,c,h,w = list(x.size())
		results = []
		encoded_reps,rec = self.segnets[0](x[:,0])	#Assuming x is a batch,num_ip,c,h,w tensor
		rec = rec.unsqueeze(1)
		for i in range(1,self.num_inputs):
			enc,rec1 = self.segnets[i](x[:,i])
			encoded_reps = torch.cat((encoded_reps,enc),1)	#nchw
			rec = torch.cat((rec,rec1.unsqueeze(1)),1)	

		encoded_reps_rec,meansr,meansumr = self.attention_gates_rec(encoded_reps)
		encoded_reps_pred,meansp,meansump = self.attention_gates_pred(encoded_reps)

		rec_im = torch.reshape(self.reconstruct_image(encoded_reps_rec),[batch,c,h,w])	#get batch etc
		pred_im = self.predict_label(encoded_reps_pred)

		return (rec,rec_im,pred_im,meansr,meansumr,meansp,meansump)

