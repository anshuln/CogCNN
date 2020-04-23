#Downloads and modifies the OpenImagesV6 dataset for our use 
import pandas as pd
import urllib
from tqdm import tqdm
import os
import cv2
import numpy as np
train_segmentation_dir = 'train-masks-3' 

train_dir = 'train_images'

seg_file_csv = 'train-annotations-object-segmentation.csv'
im_file_csv  = 'train-images-boxable-with-rotation.csv'
lab_file_csv = 'labels.csv'

def generate_pandas(names,seg_file,im_file,label_file):
    segs = pd.read_csv(seg_file).drop(['BoxID','BoxXMin','BoxXMax','BoxYMin','BoxYMax','PredictedIoU','Clicks'],axis=1)
    ims  = pd.read_csv(im_file).drop(['OriginalLandingURL','License','AuthorProfileURL','Author','Title','OriginalSize','OriginalMD5','Thumbnail300KURL','Rotation'],axis=1)
    labels = pd.read_csv(label_file)["LabelHex"].tolist()
    ims  = ims.merge(segs,on='ImageID')
    ims  = ims.loc[ims.MaskPath.isin(names),:]
    ims  = ims.loc[ims.LabelName.isin(labels)]
    return ims

def download_images(img_df,train_dir,counter=10000000000):
    #Step 1 - create folders having labels
    labels = set(img_df['LabelName'].tolist())
    labels_counts = dict([(x,0) for x in labels])
    for l in labels:
        os.makedirs('{}/{}'.format(train_dir,l.replace('/','#')),exist_ok=True)

    #Step 2 - download images
    index = 0
    for ind in tqdm(img_df.index): 
        lab = True
        for l in labels_counts.keys():
            lab = lab & (labels_counts[l]>counter)
        if lab:
            break
        if labels_counts[img_df['LabelName'][ind]] > counter:
            continue
        # if ind<50:
        try:
            url  = img_df['OriginalURL'][ind]
            path = '{}/{}/{}.jpg'.format(train_dir,img_df['LabelName'][ind].replace('/','#'),img_df['ImageID'][ind])  #All images are jpg
            urllib.request.urlretrieve(url,path)
            labels_counts[img_df['LabelName'][ind]] += 1
        except:
            print("HTTP Error URL doesn't exist")
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def segment_images(img_df,train_dir,train_segmentation_dir):
    for ind in img_df.index: 
        # if ind<50:
        try:
            impath = '{}/{}/{}.jpg'.format(train_dir,img_df['LabelName'][ind].replace('/','#'),img_df['ImageID'][ind])  #All images are jpg
            seg_path = "{}/{}".format(train_segmentation_dir,img_df['MaskPath'][ind])
            image = cv2.imread(impath)#/255.0
            if image is None:
                print("File doesn't exist") 
                continue
            image = image/255.0
            imshape = (image.shape[1],image.shape[0])
            segmask = cv2.resize(cv2.imread(seg_path),imshape)/255.0
            image = image*segmask
            rmin,rmax,cmin,cmax = bbox(image)
            image = image[rmin:rmax,cmin:cmax,:]
            impath  = '{}/{}/{}$segmented.jpg'.format(train_dir,img_df['LabelName'][ind].replace('/','#'),img_df['ImageID'][ind])
            cv2.imwrite(impath,(image)*255.0)
            print("Wrote file into {}".format(impath))
        except FileNotFoundError:
            print("File doesn't exist") 

if __name__ == "__main__":
    mask_paths = []
    for file in os.listdir(train_segmentation_dir):
        mask_paths.append(file.split('/')[-1])

    img_df = generate_pandas(mask_paths,seg_file_csv,im_file_csv,lab_file_csv)
    print("_______DOWNLOADING IMAGES____________")
    download_images(img_df,train_dir,5)
    print("__________DOWLOAD COMPLETE___________")
    segment_images(img_df,train_dir,train_segmentation_dir)
