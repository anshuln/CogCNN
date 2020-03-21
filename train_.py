import torch 

from model_ import MultiTaskModel
from skimage import io

num_classes = 31
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(num_classes).to(device)




class TrainingDataset(torch.utils.data.Dataset):

	def __init__(self,list_of_paths,stream_dict):
		"""
		Args:
			text_file(string): path to text file
			root_dir(string): directory with all train images
		"""
		self.file_paths = list_of_paths
		self.stream_dict = stream_dict

	def __len__(self):
		return len(self.list_of_paths)

	def __getitem__(self, idx):
		file_path = self.file_paths[idx]
		paths_to_read = []
		curr_path = file_path.split('/')
		for p in self.stream_dict.keys():
			path = [x for x in curr_path]
			path[-1] = self.stream_dict[p]+curr_path[-1][1:]
			path[5] = p
			paths_to_read.append('/'.join(path))
		label_path = [x for x in curr_path]
		label_path[-1] = 'm' + label_path[-1][1:]
		label_path[5] = 'fruit_main'

		vect = io.imread(paths_to_read[0])
		vect = np.expand_dims(vect,0)
		for path in paths_to_read[1:]:
			img = io.imread(path)
			vect = np.concat([vect,np.expand_dims(img,0)],0)
		label_img = io.imread(label_path)
		label = curr_path[6]	#convert to one hot
		sample = {'image': vect, 'label_img': label_img,'label':label}
		return sample
# mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
#                                    root_dir = 'Downloads/mnist_m/mnist_m_train')

# mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

# Loss and optimizer
criterion_pred = nn.CrossEntropyLoss()
criterion_rec  = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, rec, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		rec    = rec.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion_rec(outputs[1], rec) + criterion_rec(outputs[0]+images)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#TODO put all this in a function
for i in range(num_inputs):
	model.segnets[i].set_train_false()	#TODO write this function
model.attention_gates_rec.set_train_false()

for p in model.reconstruct_image.parameters():
	p.requires_grads = False


for epoch in range(num_epochs):
	for i, (images, rec, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		rec    = rec.to(device)

		# Forward pass
		outputs = model(images)

		loss = criterion_pred(nn.Softmax(outputs[2]),labels)  #+ Add regularizer   
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
