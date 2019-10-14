import cv2
import pandas as pd
import numpy as np 
import os
import h5py

def get_labels(folder):
	#Returns all labels in the training set
	labels = []
	for _, f, _ in os.walk(folder):
		labels += (f)
	return list(set(labels))

def generate_h5_files(train_file_dir_list=['greyscale'],test_file_dir='edges'):
	base_dir = train_file_dir_list[0]
	labels = get_labels(test_file_dir) 
	idx_img = 0
	idx_h5 = 0
	img_batch = [[]]*len(train_file_dir_list)	# TODO shuffle training data
	labels_batch = [[]]*2
	batch_size = 50
	for dirname, _, filenames in os.walk(base_dir):
		for filename in filenames:
			if ".jpg" in filename:
				for dir_idx,dir in enumerate(train_file_dir_list):
					img_batch[dir_idx].append(cv2.imread(os.path.join(dirname.replace(base_dir,dir), filename)))
				# print(dirname,filename)
				labels_batch[0].append(cv2.imread(os.path.join(dirname.replace(base_dir,test_file_dir), filename))) 
				labels_batch[1].append(labels.index(dirname.split('/')[-1]))
				idx_img+=1
				if ((idx_img + 1) % batch_size == 0):# or idx_img == len(images) - 1:
					#TODO save h5 files
					filename_h5_data = 'data/' + '%d.npy' % (idx_h5)
					filename_h5_label = 'label/' + '%d.npy' % (idx_h5)
					print('Saving {}.npy...'.format(idx_h5))
					# filelist_h5.write('./%d.h5\n' % (idx_h5))
					np.save(filename_h5_data,np.array(img_batch))
					np.save(filename_h5_label,np.array(labels_batch))
					# file = h5py.File(filename_h5, 'w')
					# file.create_dataset('data', data=np.concatenate(img_batch,axis=0))
					# file.create_dataset('label', data=np.concatenate(labels_batch,axis=0))
					# file.close()
					img_batch = [[]]*len(train_file_dir_list)	# TODO shuffle training data
					labels_batch = [[]]*2
					idx_h5 = idx_h5 + 1


generate_h5_files()