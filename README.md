# BinaryClassification_ADR
Automatic binary classification of adverse drug reaction mentioning posts
link: https://healthlanguageprocessing.org/sharedtask2/

# Data description(additional dataset not used this time):
  1. Number of sentences:	6761
  2. Number of sentences of label 0:	6034
  3. Number of sentences of label 1:	727
  4. Vocab size:	13657
  5. Max sentence length:	40
# build Model:
1. Embedding layer
	1. Convert words to vectors based on word2vec
	2. One channel
	3. set embedding dimension to 300

2. Convolution layer and pooling for each filter
	1. I use three different widths for filters [3, 4, 5], 100 for each filter
	2. padded input to 48(vector_length, max sentence length 40)
	3. convolution operation based on filter and inputs from embedding layer 
	4. Maxpooling over the outputs of convolution layer
	5. combine all the pooled features 3 types of filter and 100 filters for each type

3. add dropout
	set dropout_keep_prob as 0.5 when we start train the model

4. set lost function
	we use cross entropy to define lost function of this model

5. set optimizer
	I use AdamOptimizer to minimize the lost function as we set the learning rate as 0.01 at beginning

# train model
	1. Convert each sentence into word vector with dimension of (48 * 300) using word2vec
	2. split all posts to two different files according to labels
	3. split data to train dataset(90%, unbalanced) and test dataset(10%)
	4. select same size of data from label 1 and label 0 to make a balanced training data. Repeat this process for 9 times(ratio of label0 to label1)
	5. Set epoch as "25"
	6. I use batch of size 100 to train model 
# about running model
1) Clone the repository recursively to get all folder and subfolders  
2) Download Google's word embeddings binary file from [https://code.google.com/p/word2vec/](https://code.google.com/archive/p/word2vec/) extract it, and place it under `data/` folder
3) Using the command in the home directory of this repository
		`python driver.py`
