# CogCNN
Official implementation of the paper CogCNN: Mimicking Human Cognition to Resolve Shape Texture Bias presented at the [BAICS workshop at ICLR 2020](https://baicsworkshop.github.io/).

## Instructions to run-
* Have the following directory structure for data - 
	multitask/
	|->amazon\_silhouette
	  |->(Folders corresponding to labels)
	|->amazon\_texture
	  |->images
		 |->(Folders corresponding to labels)
	|->greyscale
	   |->(Folders corresponding to labels)
	|->edges
	   |->(Folders corresponding to labels)
	|->data
	|->label
	   |->labels
	   |->images
	|->Reconstructed_Results
* Run `generate_dataset.py` if the `data` and `label` dir do not contain `.npy` files
* Run `train.py`, the reconstructed images will be in Reconstructed_Results dir	
