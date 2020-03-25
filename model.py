import torch 
from torch import nn 

class MultiTaskModel(nn.Module):
    def __init__(self,image_shape,num_labels,num_inputs=4):
        super(MultiTaskModel,self).__init__()
        self.image_shape = image_shape 
        self.num_inputs = num_inputs
        self.num_labels = num_labels

        self.segnets = []

        for i in range(num_inputs):
            self.segnets.append(SegNet().cuda())
        
        self.attention_gates_rec  = CompleteAttention([nn.Conv2d(512*num_inputs,128,3,padding=2),nn.ReLU(),nn.Conv2d(128,512*num_inputs,3,padding=0),nn.Sigmoid()],4) 
        self.attention_gates_pred = CompleteAttention([nn.Conv2d(512*num_inputs,128,3,padding=2),nn.ReLU(),nn.Conv2d(128,512*num_inputs,3,padding=0),nn.Sigmoid()],4)

        self.reconstruct_image = nn.Sequential(Flatten(),nn.BatchNorm1d(512*4*16),nn.Linear(512*4*16,1024),nn.Linear(1024,image_shape[0]*image_shape[1]*image_shape[2],nn.Sigmoid())) #Add sigmoid

        self.predict_label = nn.Sequential(nn.Conv2d(512*num_inputs,128,2,padding=0),nn.ReLU(),nn.Conv2d(128,128,2),Flatten(),nn.Linear(512,num_labels
            ))
    def forward(self,x):
#         print(x.size())
        batch,num_ip,c,h,w = list(x.size())
        results = []
        encoded_reps,rec = self.segnets[0](x[:,0])  #Assuming x is a batch,num_ip,c,h,w tensor
        rec = rec.unsqueeze(1)
        for i in range(1,self.num_inputs):
            enc,rec1 = self.segnets[i](x[:,i])
            encoded_reps = torch.cat((encoded_reps,enc),1)  #nchw
            rec = torch.cat((rec,rec1.unsqueeze(1)),1)  

        encoded_reps_rec,meansr,meansumr = self.attention_gates_rec(encoded_reps)
        encoded_reps_pred,meansp,meansump = self.attention_gates_pred(encoded_reps)

        rec_im = self.reconstruct_image(encoded_reps_rec).view(batch,c,h,w)  #get batch etc
        pred_im = self.predict_label(encoded_reps_pred)

        return (rec,rec_im,pred_im,meansr,meansumr,meansp,meansump)
    def get_attention_stats(self,x):
        batch,num_ip,c,h,w = list(x.size())
        results = []
        encoded_reps,rec = self.segnets[0](x[:,0])  #Assuming x is a batch,num_ip,c,h,w tensor
        rec = rec.unsqueeze(1)
        for i in range(1,self.num_inputs):
            enc,rec1 = self.segnets[i](x[:,i])
            encoded_reps = torch.cat((encoded_reps,enc),1)  #nchw
            rec = torch.cat((rec,rec1.unsqueeze(1)),1)  

        encoded_reps_rec,meansr,meansumr = self.attention_gates_rec(encoded_reps)
        encoded_reps_pred,meansp,meansump = self.attention_gates_pred(encoded_reps)
        
        return meansr,meansp,meansumr,meansump
