#TODO refactor to get argparse for models and dataset
from model_bits_and_pieces import MultiTaskModel
from tqdm import tqdm

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np

import pickle
import os
import sys
import argparse

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

sys.setrecursionlimit(10**6)

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
            l,l_all = model.validate_batch(input_batch,label_im_batch,label_batch)
            v_acc += l 
            sample = np.random.randint(15)
            check = input_batch[:,sample:sample+1]
            res = model.call(check)
            fig, ax = plt.subplots(3, 3)
            # print("Edges",np.max(res[3][0].numpy()),np.min(res[3][0].numpy()))
            # print("Silhuette",np.max(res[0][0].numpy()),np.min(res[0][0].numpy()))
            # print("Greyscale",np.max(res[2][0].numpy()),np.min(res[2][0].numpy()))
            vloss.append([l.numpy() for l in l_all])
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
            try:
                fig, ax = plt.subplots(3, 5)
                aten_rec,aten_pred = model.getAttentionMap(check)
                for i in range(4):
                    ax[0,i].imshow(aten_rec[i,0].mean(axis=0).reshape((32,32)))
                    ax[1,i].imshow(aten_pred[i,0].mean(axis=0).reshape((32,32)))
                    ax[2,i].imshow(res[i][0])
                ax[0,4].imshow(label_im_batch[sample])
                ax[1,4].imshow(res[4][0])
                plt.savefig("AttentionMaps/{}-{}.png".format(epoch,f))
                plt.close()


            except:
                pass        

        log.write("Epoch - {} Training_Loss - {} validation_loss - {},val_acc - {}".format(epoch,np.array(tloss).mean(axis=0),np.array(vloss).mean(axis=0),v_acc/(len(val_files)*16)))
        log.write("\n")
        print("Val-loss {}".format(np.array(vloss).mean(axis=0)[-1]))
        if epoch % 5 == 0:
            model.save('trained_models')

if __name__ == "__main__":
    TF_CPP_MIN_LOG_LEVEL = 4
    parser = argparse.ArgumentParser()

    #-db DATABSE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-a","--attention", help="Using attention or not",action="store_true")
    parser.add_argument("-p","--pix2pix", help="Using pix2pix",action="store_true")
    parser.add_argument("-t","--twostage", help="Using two stage",action="store_true")
    parser.add_argument("-s","--small", help="Small dataset",action="store_true")

    args = parser.parse_args()

    # if(len(sys.argv)==2):
    #   trainableVariablesFile = sys.argv[1]
    #   trainableVariables = pickle.load(open(trainableVariablesFile,'rb'))
    #   model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels,trainableVariables=trainableVariables)
    # else:
    #   model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels)

    if args.small:
        train_dir = 'data_small/'
        label_im_dir = 'label_small/images'
        label_dir = 'label_small/labels'
    model = MultiTaskModel(num_inputs=num_inputs,image_shape=image_shape,num_labels=num_labels,attention=args.attention,pix2pix=args.pix2pix,two_stage=args.twostage)
    
    d1 = np.load("{}/0v.npy".format(train_dir),allow_pickle=True)
    print(d1[:,:1].shape)
    print("Building Model...")
    model.build(d1[:,:1])
    
    # fig, ax = plt.subplots(3, 3)
    # # print(res[0][0])
    # # try:
    # ax[0,0].imshow(res[0][0])
    # ax[1,0].imshow(res[1][0])
    # ax[2,0].imshow(res[2][0])
    # ax[0,1].imshow(res[3][0])
    # ax[1,1].imshow(res[4][0])
    # # ax[2,1].imshow(label_im_batch[sample])
    # ax[2,2].imshow(d1[:,0][0])
    # ax[0,2].imshow(d1[:,0][1])
    # ax[1,2].imshow(d1[:,0][2])
    # plt.savefig("Reconstructed_Results/Init.png")
    # plt.close()
    # for x in res:
    #   print("Shape",x.shape)  
    print("Model built")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.setTrainableVariables()
    train(model,optimizer,100)
    # pred = model.call(d1)



