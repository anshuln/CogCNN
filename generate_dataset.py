import cv2
import pandas as pd
import numpy as np 
import os
import h5py
from random import shuffle

def get_labels(folder):
	#Returns all labels in the training set
	labels = []
	for _, f, _ in os.walk(folder):
		labels += (f)
	return list(set(labels))

def one_hot(label_index,len_labels):
	one_hot = [0]*len_labels
	one_hot[label_index] = 1
	return one_hot
def generate_npy_files(train_file_dir_list=['amazon_silhouette','amazon_texture/images','edges','greyscale'],test_file_dir='images'):
	base_dir = train_file_dir_list[0]
	labels = get_labels(test_file_dir) 
	idx_img = 0
	idx_h5 = 0
	img_batch = [list() for i in range(len(train_file_dir_list))]   # TODO shuffle training data
	labels_batch = []
	labels_images_batch = []
	batch_size = 16
	img_size = (64,64)
	image_paths = []
	for l in labels:
		dir = "{}/{}".format(base_dir,l)
		for file in os.listdir(dir):
			if file.endswith(".jpg"):
				image_paths.append("{}/{}".format(l,file))
	shuffle(image_paths)
	print(labels)
	for path in image_paths:
		for dir_idx in range(len(train_file_dir_list)):
			dire = train_file_dir_list[dir_idx]
			img_batch[dir_idx].append(cv2.resize((cv2.imread("{}/{}".format(dire,path))/255.0),img_size))
			#print(len(img_batch),len(img_batch[0]),len(img_batch[1]))
			#print(dir_idx,"{}/{}".format(dire,path))
		labels_images_batch.append(cv2.resize((cv2.imread("{}/{}".format(test_file_dir,path))/255.0),img_size))
		labels_batch.append(one_hot(labels.index(path.split('/')[0]),len(labels)))
		idx_img+=1
		if ((idx_img + 1) % batch_size == 0):# or idx_img == len(images) - 1:
			#TODO save h5 files
			# if idx_h5 > 45:
			#   return
			if idx_h5 < 10:
				filename_h5_data = 'data/' + '%dv.npy' % (idx_h5)
				filename_h5_label = 'label/labels/' + '%dv.npy' % (idx_h5)
				filename_h5_label_images = 'label/images/' + '%dv.npy' % (idx_h5)
			else:
				filename_h5_data = 'data/' + '%d.npy' % (idx_h5)
				filename_h5_label = 'label/labels/' + '%d.npy' % (idx_h5)
				filename_h5_label_images = 'label/images/' + '%d.npy' % (idx_h5)

			print('Saving {}.npy...'.format(idx_h5))
			# filelist_h5.write('./%d.h5\n' % (idx_h5))
			np.save(filename_h5_data,np.array(img_batch))
			np.save(filename_h5_label,np.array(labels_batch))
			np.save(filename_h5_label_images,np.array(labels_images_batch))
			# file = h5py.File(filename_h5, 'w')
			# file.create_dataset('data', data=np.concatenate(img_batch,axis=0))
			# file.create_dataset('label', data=np.concatenate(labels_batch,axis=0))
			# file.close()
			img_batch = [list() for i in range(len(train_file_dir_list))]   # TODO shuffle training data
			labels_batch = []
			labels_images_batch = []
			idx_h5 = idx_h5 + 1


	# for dirname, _, filenames in os.walk(base_dir):
	#   for filename in filenames:
	#       if ".jpg" in filename:
	#           for dir_idx,dir in enumerate(train_file_dir_list):
	#               img_batch[dir_idx].append(cv2.imread(os.path.join(dirname.replace(base_dir,dir), filename)))
	#           # print(dirname,filename)
	#           labels_batch[0].append(cv2.imread(os.path.join(dirname.replace(base_dir,test_file_dir), filename))) 
	#           labels_batch[1].append(one_hot(labels.index(dirname.split('/')[-1]),len(labels)))
	#           idx_img+=1
	#           if ((idx_img + 1) % batch_size == 0):# or idx_img == len(images) - 1:
	#               #TODO save h5 files
	#               filename_h5_data = 'data/' + '%d.npy' % (idx_h5)
	#               filename_h5_label = 'label/' + '%d.npy' % (idx_h5)
	#               print('Saving {}.npy...'.format(idx_h5))
	#               # filelist_h5.write('./%d.h5\n' % (idx_h5))
	#               np.save(filename_h5_data,np.array(img_batch))
	#               np.save(filename_h5_label,np.array(labels_batch))
	#               # file = h5py.File(filename_h5, 'w')
	#               # file.create_dataset('data', data=np.concatenate(img_batch,axis=0))
	#               # file.create_dataset('label', data=np.concatenate(labels_batch,axis=0))
	#               # file.close()
	#               img_batch = [[]]*len(train_file_dir_list)   # TODO shuffle training data
	#               labels_batch = [[]]*2
	#               idx_h5 = idx_h5 + 1


generate_npy_files()

