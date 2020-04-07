import torch 

from model_ import MultiTaskModel
from skimage import io

num_classes = len(labels_list)
num_inputs = 4
image_shape = (3,64,64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(image_shape,num_classes).to(device)

model.cuda()
num_epochs = 2



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
        for p in self.stream_dict:
            path = [x for x in curr_path]
            path[-1] = p[1]+curr_path[-1][1:]
            path[6] = p[0]
            paths_to_read.append('/'.join(path))
        label_path = [x for x in curr_path]
        label_path[-1] = 'm' + label_path[-1][1:]
        label_path[6] = 'fruit_main'
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
        label = np.array(labels_list.index(curr_path[7]))    #convert to one hot
        sample = {'image': torch.tensor(vect,dtype=torch.float32), 'label_img': torch.tensor(label_img,dtype=torch.float32),'label':torch.tensor(label)}
        return sample
# mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
#                                    root_dir = 'Downloads/mnist_m/mnist_m_train')

# mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

# Loss and optimizer
transformed_dataset = TrainingDataset(glob.glob('/kaggle/input/fruitsmulti/fruit/*/Training/*/*'),
                               [('fruit_texture','t'),('fruit_shape','s'),('fruit_edge','e')])

transformed_test_ds = TrainingDataset(glob.glob('/kaggle/input/fruitsmulti/fruit/*/Test/*/*'),
                               [('fruit_texture','t'),('fruit_shape','s'),('fruit_edge','e')])

train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=16,
                        shuffle=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(transformed_dataset, batch_size=100,
                        shuffle=True, num_workers=4)
criterion_pred = nn.CrossEntropyLoss()
criterion_rec  = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
#         print(i+1)
        images = sample["image"].to(device)
#         labels = sample["label"].to(device)
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
#             print(np.moveaxis(outputs[1].cpu().detach().numpy()[0],0,2).shape,np.moveaxis(rec.cpu().detach().numpy()[0],0,2).shape)
            cv2.imwrite('{}-{}-rec.jpg'.format(epoch+1,i+1),np.moveaxis(outputs[1].cpu().detach().numpy()[0],0,2)*255.0)
            cv2.imwrite('{}-{}-act.jpg'.format(epoch+1,i+1),np.moveaxis(rec.cpu().detach().numpy()[0],0,2)*255.0)
            torch.save(model.state_dict(),"saved-model")

#TODO put all this in a function
for i in range(num_inputs):
    model.segnets[i].set_train_false()  #TODO write this function
model.attention_gates_rec.set_train_false()

for p in model.reconstruct_image.parameters():
    p.requires_grads = False


for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        images = sample["image"].to(device)
        labels = sample["label"].to(device)
        rec    = sample["label_img"].to(device)
        # Forward pass
        outputs = model(images)

#         print(.size(),labels.size())
        loss = criterion_pred(outputs[2],labels)  #+ Add regularizer   
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            with torch.no_grad():
                accuracy = torch.zeros((1,))
                ind = 0
                for sample in test_loader:
                    if ind>=2:
                        break
#                     sample   = test_loader[(i+1)/100]
                    im_test  = sample["image"].to(device)
                    lab_test = sample["label"].to(device)
                    out_test  = model(im_test)
                    accuracy += torch.sum(torch.argmax(out_test[2],1)==lab_test)
                    ind+=1
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(),accuracy.item()))
            torch.save(model.state_dict(),"saved-model")
            


# with torch.no_grad():
metrics_rec  = np.zeros((num_classes,num_inputs))
metrics_pred = np.zeros((num_classes,num_inputs))
counts = np.zeros((num_classes,1))+1e-9
for j,sample in enumerate(train_loader):
    images = sample["image"].to(device)
    labels = sample["label"].to(device)
    rec    = sample["label_img"].to(device)
#     print(images.size())
#     print(model.get_attention_stats(images))
#     break
#   for f in (train_files):
    for i in range(images.size()[0]):
        label = labels[i]
        mr,mp,msr,msp = model.get_attention_stats(images[i:i+1])
        # print(aten_rec[:,0].shape)
        metrics_rec[label] += np.array([x.detach().cpu().numpy() for x in mr])/msr.detach().cpu().numpy()
        metrics_pred[label]+= np.array([x.detach().cpu().numpy() for x in mp])/msp.detach().cpu().numpy()
        counts[label] += 1
#         break
#     break
print(metrics_rec/counts)
print(metrics_pred/counts)
