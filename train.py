from model import MultiTaskModel
from tqdm import tqdm

import tensorflow as tf
import numpy as np

import pickle
import os

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
#tf.config.experimental.set_memory_growth(physical_devices[1], True)
train_dir = 'data/'
label_im_dir = 'label/images'
label_dir = 'label/labels'
log_file = 'log.txt'

num_inputs = 4
image_shape = (64,64,3)
num_labels = 31


def train(model,optimizer,epochs):  #TODO add validation, generator for datasets
    train_files = []
    val_files = []
    for file in os.listdir(train_dir):
        if file.endswith(".npy"):   #Change to h5 later
            train_files.append(file)
    for file in os.listdir(train_dir):
        if file.endswith("v.npy"):   #Change to h5 later
            val_files.append(file)
    train_files = [x for x in train_files if x not in val_files]
    log = open(log_file,"w")
    for epoch in range(epochs):
        print("Epoch num {}".format(epoch))
        tloss = []
        vloss = []
        v_acc = 0
        for f in tqdm(train_files):
            input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
            label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
            label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
            l,l_all = model.train_on_batch(input_batch,label_im_batch,label_batch,optimizer)
            tloss.append([l.numpy() for l in l_all])
        for f in val_files:
            input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
            label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
            label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
            l_all = model.validate_batch(input_batch,label_im_batch,label_batch)
            # v_acc += l 
            sample = np.random.randint(15)
            check = input_batch[:,sample:sample+1]
            res = model.call(check)
            fig, ax = plt.subplots(3, 3)
            print("Edges",np.max(res[3][0].numpy()))
            print("Silhuette",np.max(res[0][0].numpy()),np.min(res[0][0].numpy()))
            print("Greyscale",np.max(res[2][0].numpy()),np.min(res[2][0].numpy()))
            try:
	            ax[0,0].imshow(res[0][0])
	            ax[1,0].imshow(res[1][0])
	            ax[2,0].imshow(res[2][0])
	            ax[0,1].imshow(res[3][0])
	            ax[1,1].imshow(res[4][0])
	            ax[2,1].imshow(label_im_batch[sample])
	            ax[2,2].imshow(input_batch[:,sample][0])
	            ax[0,2].imshow(input_batch[:,sample][1])
	            ax[1,2].imshow(input_batch[:,sample][2])
	            plt.savefig("Reconstructed_Results/{}-{}.png".format(epoch,f))
	            plt.close()
            except:
                pass	
            vloss.append([l.numpy() for l in l_all])

        log.write("Epoch - {} Training_Loss - {} validation_loss - {}".format(epoch,np.array(tloss).mean(axis=0),np.array(vloss).mean(axis=0)))
        log.write("\n")
        norms = model.getWeightNorms()
        print(norms)
        log.write("Norm_rec - {}, Norm_pred - {}".format(norms[1],norms[0]))
        log.write("\n")
        print("Val-loss {}",.format(np.array(vloss).mean(axis=0)[-1]))
        # if epoch % 5 == 0:
        #     pickle.dump(model.trainableVariables,open("trained_models/model_{}.pkl".format(epoch),"wb"))

if __name__ == "__main__":
    TF_CPP_MIN_LOG_LEVEL = 4
    model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels)
    d1 = np.load("{}/0v.npy".format(train_dir),allow_pickle=True)
    print(d1[:,:1].shape)
    print("Building Model...")
    res = model.call(d1[:,:1])
    fig, ax = plt.subplots(3, 3)
    # print(res[0][0])
    # try:
    ax[0,0].imshow(res[0][0])
    ax[1,0].imshow(res[1][0])
    ax[2,0].imshow(res[2][0])
    ax[0,1].imshow(res[3][0])
    ax[1,1].imshow(res[4][0])
    # ax[2,1].imshow(label_im_batch[sample])
    ax[2,2].imshow(d1[:,0][0])
    ax[0,2].imshow(d1[:,0][1])
    ax[1,2].imshow(d1[:,0][2])
    plt.savefig("Reconstructed_Results/Init.png")
    plt.close()
    # except:
    # 	pass	
    for x in res:
        print("Shape",x.shape)  
    print("Model built")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.setTrainableVariables()
    train(model,optimizer,1000)
    # pred = model.call(d1)



