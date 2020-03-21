#TODO refactor to get argparse for models and dataset
from model import MultiTaskModel
from tqdm import tqdm

# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np

import pickle
import os
import sys
import argparse

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
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

def generate_class_wise_metric(model,train_files):
    metrics_rec  = np.zeros((num_labels,num_inputs))
    metrics_pred = np.zeros((num_labels,num_inputs))
    counts = np.zeros((num_labels,1))
    for f in (train_files):
        input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
        label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
        for i in range(input_batch.shape[1]):
            label = tf.math.argmax(label_batch[i],axis=0)
            aten_rec,aten_pred,prediction = model.getAttentionMap(input_batch[:,i:i+1])
            # print(aten_rec[:,0].shape)
            metrics_rec[label] += aten_rec[:,0].mean(axis=(1,2,3))
            metrics_pred[label]+= aten_pred[:,0].mean(axis=(1,2,3))
            counts[label] += 1
    print(metrics_rec/counts)
    print(metrics_pred/counts)

def train(model,optimizer,epochs,two_stage=False):  #TODO add validation, generator for datasets
    train_files = []
    val_files = []
    for file in os.listdir(train_dir):
        if file.endswith(".npy"):   #Change to h5 later
            train_files.append(file)
    for file in os.listdir(train_dir):
        if file.endswith("v.npy"):   #Change to h5 later
            val_files.append(file)
    train_files = [x for x in train_files if x not in val_files]
    max_epochs_rec = epochs
    epochs = 2*epochs if two_stage else epochs
    for epoch in range(epochs):
        print("Epoch num {}".format(epoch))
        tloss = []
        vloss = []
        v_acc = 0

        for f in tqdm(train_files):
            input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
            label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
            label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
            classification = False
            if epoch>=max_epochs_rec:
                classification = True
            l,l_all = model.train_on_batch(input_batch,label_im_batch,label_batch,optimizer,classification=classification)
#           print(l.numpy())
            tloss.append([l.numpy() for l in l_all])

        log_aten = open('log_aten.txt','a')
        log_aten.write("Epoch - {}".format(epoch))
        log_aten.write("\n-------------------\n")
        log_aten.close()
        for f in val_files:
            input_batch = np.load("{}/{}".format(train_dir,f),allow_pickle=True)
            label_batch = np.load("{}/{}".format(label_dir,f),allow_pickle=True)
            label_im_batch = np.load("{}/{}".format(label_im_dir,f),allow_pickle=True)
            l,l_all = model.validate_batch(input_batch,label_im_batch,label_batch)
            v_acc += l 
            sample = np.random.randint(15)
            check = input_batch[:,sample:sample+1]
            res = model.call(check)
            # print("Edges",np.max(res[3][0].numpy()),np.min(res[3][0].numpy()))
            # print("Silhuette",np.max(res[0][0].numpy()),np.min(res[0][0].numpy()))
            # print("Greyscale",np.max(res[2][0].numpy()),np.min(res[2][0].numpy()))
            vloss.append([l.numpy() for l in l_all])
            # print(l_all[-1].numpy().shape)
            if epoch % 5 == 0:
                # fig, ax = plt.subplots(3, 3)
                # try:
                #   ax[0,0].imshow(res[0][0])
                #   ax[1,0].imshow(res[1][0])
                #   ax[2,0].imshow(res[2][0])
                #   ax[0,1].imshow(res[3][0])
                #   ax[1,1].imshow(res[4][0])
                #   ax[2,1].imshow(label_im_batch[sample])
                #   ax[2,2].imshow(input_batch[:,sample][0])
                #   ax[0,2].imshow(input_batch[:,sample][1])
                #   ax[1,2].imshow(input_batch[:,sample][2])
                #   plt.savefig("Reconstructed_Results/{}-{}.png".format(epoch,f))
                #   plt.close()
                # except:
                #   pass    
                try:
                    aten_rec,aten_pred,prediction = model.getAttentionMap(check)
                    if model.attention == 'self':
                        fig, ax = plt.subplots(3, 5)
                        # print(aten_rec.shape)
                        log_aten = open('log_aten.txt','a')
                        log_aten.write("Reconstruction - ")
                        for i in range(4):
                            ax[0,i].imshow(aten_rec[i,0].mean(axis=0).reshape((32,32)))
                            log_aten.write("{} ".format(aten_rec[i,0].mean()))
                        log_aten.write("\nPrediction - ")
                        for i in range(4):
                            ax[1,i].imshow(aten_pred[i,0].mean(axis=0).reshape((32,32)))
                            log_aten.write("{} ".format(aten_pred[i,0].mean()))
                            ax[2,i].imshow(res[i][0])
                        log_aten.write("\n Label given - {}".format(prediction))
                        log_aten.write("\n==\n")
                        log_aten.close()
                        ax[0,4].imshow(label_im_batch[sample])
                        ax[1,4].imshow(res[4][0])
                        ax[2,4].text(30,30,prediction)
                        plt.savefig("{}-{}.png".format(epoch,f))
                        plt.close()
                    elif model.attention == 'multi':
                        fig, ax = plt.subplots(2)
                        # print(aten_rec.shape)
                        # log_aten = open('AttentionMaps/log_aten.txt','a')
                        # log_aten.write("Reconstruction - ")
                        # for i in range(2):
                        ax[0].imshow(aten_rec.reshape((256,32)))
                            # log_aten.write("{} ".format(aten_rec[i,0].mean()))
                        # log_aten.write("\nPrediction - ")
                        # for i in range(4):
                        ax[1].imshow(aten_pred.reshape((256,32)))
                            # log_aten.write("{} ".format(aten_pred[i,0].mean()))
                            # ax[2,i].imshow(res[i][0])
                        # log_aten.write("\n Label given - {}".format(prediction))
                        # log_aten.write("\n==\n")
                        # log_aten.close()
                        # ax[2,4].text(30,30,prediction)
                        plt.savefig("{}-{}.png".format(epoch,f))
                        plt.close()
                    fig, ax = plt.subplots(2)
                    ax[0].imshow(label_im_batch[sample])
                    ax[1].imshow(res[4][0])
                    plt.savefig("{}-{}-rec.png".format(epoch,f))
                    plt.close()


                except:
                    assert False
                    pass        
        log = open(log_file,"a")
        print(np.array(vloss).shape)
        log.write("Epoch - {} Training_Loss - {} validation_loss - {},val_acc - {}".format(epoch,np.array(tloss).mean(axis=0),np.array(vloss).mean(axis=0),v_acc/(len(val_files)*16)))
        log.write("\n")
        log.close()
        print("Val-loss {}".format(np.array(vloss).mean(axis=0)[-1]))
        if epoch % 5 == 0:
            model.save('trained_models')
    # generate_class_wise_metric(model,train_files)


if __name__ == "__main__":
    TF_CPP_MIN_LOG_LEVEL = 4
    parser = argparse.ArgumentParser()

    #-db DATABSE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-a","--attention", help="Using attention or not",choices=['self','multi'],default=None)
    parser.add_argument("-p","--pix2pix", help="Using pix2pix",action="store_true")
    parser.add_argument("-t","--twostage", help="Using two stage",action="store_true")
    parser.add_argument("-s","--small", help="Small dataset",action="store_true")
    parser.add_argument("-e","--epochs",help="Num of epochs",type=int, default=50)

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
    log = open(log_file,'w')
    log.close()
    log = open('AttentionMaps/log_aten.txt','w')
    log.close()
    train(model,optimizer,args.epochs,args.twostage)
    # pred = model.call(d1)



