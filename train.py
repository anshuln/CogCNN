import torch 

from model_ import MultiTaskModel
from skimage import io

num_classes = 31
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(num_classes).to(device)





num_classes = 31
image_shape = (3,64,64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(image_shape,num_classes).to(device)

model.cuda()
num_epochs = 5



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
		return len(self.file_paths)

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
		label_path = '/'.join(label_path)
		
		vect = cv2.resize(cv2.imread(paths_to_read[0]),(64,64))/255.0
		vect = np.expand_dims(np.moveaxis(vect,2,0),0)
		for path in paths_to_read[1:]:
			img = cv2.resize(cv2.imread(path),(64,64))/255.0
			if len(img.shape) == 3:
				img = np.expand_dims(np.moveaxis(img,2,0),axis=0)
			else:
				img = np.expand_dims(np.expand_dims(img,axis=0),axis=1)
#             print(img.shape)
			vect = np.concatenate([vect,img],0)
		label_img = cv2.resize(cv2.imread(label_path),(64,64))/255.0
		label_img = np.moveaxis(label_img,2,0)
		label = np.array(one_hot(labels.index(curr_path[7]),len(labels)))    #convert to one hot
		sample = {'image': torch.tensor(vect,dtype=torch.float32), 'label_img': torch.tensor(label_img,dtype=torch.float32),'label':torch.tensor(label)}
		return sample
# mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
#                                    root_dir = 'Downloads/mnist_m/mnist_m_train')

# mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

# Loss and optimizer
transformed_dataset = TrainingDataset(glob.glob('/kaggle/input/fruitsmulti/fruit/*/Training/*/*'),
							   {'fruit_texture':'t','fruit_shape':'s','fruit_edge':'e','fruit_gray':'g'})

train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=16,
						shuffle=True, num_workers=4)
criterion_pred = nn.CrossEntropyLoss()
criterion_rec  = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, sample in enumerate(train_loader):
		images = sample["image"].to(device)
		labels = sample["label"].to(device)
		rec    = sample["label_img"].to(device)

		# Forward pass
		outputs = model(images)
#         print(outputs[1].size(),outputs[0].size(),rec.size(),images.size())
		loss = criterion_rec(outputs[1], rec) + criterion_rec(outputs[0],images)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#TODO put all this in a function
for i in range(num_inputs):
	model.segnets[i].set_train_false()  #TODO write this function
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
