# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F


def train(batch_size):
    # model.train() puts our model in train mode, which can require different
    # behavior than eval mode (for example in the case of dropout).
    model.train()
    # i is is a 1-D array with shape [batch_size]
    i = np.random.choice(train_imgs.shape[0], size=batch_size, replace=False)
    x = autograd.Variable(torch.from_numpy(train_imgs[i].astype(np.float32)))
    y = autograd.Variable(torch.from_numpy(train_lbls[i].astype(np.int)))
    optimizer.zero_grad()
    y_hat_ = model(x)
    loss = F.cross_entropy(y_hat_, y)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def accuracy(y, y_hat):
    correct = 0
    for i in range(len(y_hat)):
        y_hat_max_prob = np.argmax(y_hat[i].data.numpy())
        if(y[i]==y_hat_max_prob):
            correct += 1
    acc = correct / len(y_hat)
    return acc


def approx_train_accuracy():
    model.eval()
    idx = np.random.randint(45000,size=1000)
    acc = accuracy(train_lbls[idx], model(autograd.Variable(torch.from_numpy(train_imgs[idx,:].astype(np.float32)))))
    return acc


def val_accuracy():
    model.eval()
    acc = accuracy(dev_lbls, model(autograd.Variable(torch.from_numpy(dev_imgs.astype(np.float32)))))
    return acc

	
class ConvNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 convolution that takes in an image with one channel
        # and outputs an image with 8 channels.
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=7)
        # 3x3 convolution that takes in an image with 8 channels
        # and outputs an image with 16 channels. The output image
        # has approximately half the height and half the width
        # because of the stride of 2.
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2)
        # 1x1 convolution that takes in an image with 16 channels and
        # produces an image with 5 channels. Here, the 5 channels
        # will correspond to class scores.
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.final_conv = torch.nn.Conv2d(64, 128, kernel_size=1)
        self.linear1 = torch.nn.Linear(128, NUM_CLASSES)
    def forward(self, x):
        # Convolutions work with images of shape
        # [batch_size, num_channels, height, width]
        x = x.view(-1, HEIGHT, WIDTH).unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        n, c, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=[h, w])
        x = F.relu(self.final_conv(x).view(-1, 128))
        x = self.linear1(x)
        return x

if __name__ == "__main__":		
	images = np.array(np.load('/../data/images.npy'), dtype = np.float32)
	labels = np.array(np.load('/../data/labels.npy'),dtype = np.int16)

	images = images - images.mean()
	images = images / images.max()

	[num_images, height, width] = images.shape
	images_rs = np.reshape(images, (num_images, height * width))

			
	train_imgs = images_rs[0:45000, :]
	train_lbls = labels[0:45000]

	dev_imgs_raw = images[45000:50000, :, :]
	dev_imgs = images_rs[45000:50000, :]
	dev_lbls = labels[45000:50000]


	HEIGHT = 26
	WIDTH = 26
	NUM_CLASSES = 5
	NUM_OPT_STEPS = 2001


	model = ConvNN()
	optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-1)
	NUM_OPT_STEPS = 100001
		
	batch_size = 100
	train_accs, val_accs = [], []
	for i in range(NUM_OPT_STEPS):
		train(batch_size)
		if i % 100 == 0:
			train_accs.append(approx_train_accuracy())
			val_accs.append(val_accuracy())
			if i % 1000 == 0:
				print("%6d %5.2f %5.2f" % (i, train_accs[-1], val_accs[-1]))
				
	import matplotlib.patches as mpatches
	plt.figure(1)
	t1 = np.arange(0,len(train_accs),1)
	plt.plot(t1,train_accs,t1,val_accs)
	plt.xlabel("100 Iteration Increments")
	plt.ylabel("Accuracy")
	blue = mpatches.Patch(color='blue', label='Train')
	orange = mpatches.Patch(color='orange', label='Validation')
	plt.legend(handles=[blue, orange])


	# My starting point was a 5 layer CNN with progressively larger layer sizes.
	# My final / most successful network had the following properties:
	# -5 convolutional layers
	# -1 linear layer after the covolutional layers
	# -1 dropout layer after the first convolutional layer
	# -batch size of 100
	# -adagrad optimizer with rate of 1e-1
	# -cross entropy loss
	# -kernel sizes of 7,3,3,3,1 for the 5 convolutional layers
	# 
	# The most significant changes in terms of validation accuracy was the addition of more convolutional layers.
	# However the addition of dropout and a final linear layer decreased overfitting and increased convergance time.
	# Adding more than 5 2^n size convolutional layers to the network made the training time unbearably slow so I do not know if even more layers would improve results.
	# Adding in convolutional layers of the same size while changing the stride caused the accuracy to drop to ~20%.
	# I tried adding more than 1 linear layer to the output of the convolutional layers but that seemed to make things worse than a single linear layer.

