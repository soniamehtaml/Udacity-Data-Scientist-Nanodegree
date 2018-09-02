# Imports here
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import argparse
import time
import copy
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU if available')
parser.add_argument('--epochs', type=int, default=14, help='Number of epochs')
parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, default='Yes', help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()

with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)
	
def process_data(data_dir):
	data_dir = args.data_dir
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'
	
	# TODO: Define your transforms for the training, validation, and testing sets
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomRotation(30),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		 ]),
		'valid': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		 ]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		 ])
	}

	# TODO: Load the datasets with ImageFolder
	image_datasets = dict()
	image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
	image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
	image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	dataloaders = {
		x: data.DataLoader(image_datasets[x], batch_size = 4, shuffle = True, num_workers = 2)
		for x in list(image_datasets.keys())
	}

	return dataloaders, image_datasets

def load_model(arch=args.arch, num_labels=102, hidden_units=args.hidden_units):

    # 1. Load a pre-trained model
	if arch=='vgg19':
        # Load a pre-trained model
		model = models.vgg19(pretrained=True)
	elif arch=='alexnet':
		model = models.alexnet(pretrained=True)
	else:
		raise ValueError('Unexpected network architecture', arch)
        
    # Freeze its parameters
	for param in model.parameters():
		param.requires_grad = False
    
    # 2. Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    # Features, removing the last layer
	features = list(model.classifier.children())[:-1]
  
    # Number of filters in the bottleneck layer
	num_filters = model.classifier[len(features)].in_features

    # Extend the existing architecture with new layers
	features.extend([
		nn.Dropout(),
		nn.Linear(num_filters, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
    ])
    
	model.classifier = nn.Sequential(*features)
    
	return model

def train_model(data_dir=args.data_dir, arch=args.arch, hidden_units=args.hidden_units, epochs=args.epochs, learning_rate=args.learning_rate, gpu=args.gpu, checkpoint=args.checkpoint):  
	
	dataloaders, image_datasets = process_data(data_dir)

    # Calculate dataset sizes.
	dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    
       
	print('Network architecture:', arch)
	print('Number of hidden units:', hidden_units)
	print('Number of epochs:', epochs)
	print('Learning rate:', learning_rate)

    # Load the model     
	num_labels = len(image_datasets['train'].classes)
	model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Use gpu if selected and available
	if gpu and torch.cuda.is_available():
		print('Using GPU for training')
		device = torch.device("cuda:0")
		model.cuda()
	else:
		print('Using CPU for training')
		device = torch.device("cpu")     

                
    # Defining criterion, optimizer and scheduler
    # Observe that only parameters that require gradients are optimized
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    
			
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(epochs):
		print('Epoch {}/{}'.format(epoch + 1, epochs))
		print('-' * 10)

        # Each epoch has a training and validation phase
		for phase in ['train', 'valid']:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

            # Iterate over data.
			for inputs, labels in dataloaders[phase]:                
				inputs = inputs.to(device)
				labels = labels.to(device)
                
                # zero the parameter gradients
				optimizer.zero_grad()

                # forward
                # track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

                # statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
	model.load_state_dict(best_model_wts)
    
    # Store class_to_idx into a model property
	model.class_to_idx = image_datasets['train'].class_to_idx
	
	# Save checkpoint if requested
	if checkpoint:
		print ('Saving checkpoint to:', checkpoint) 
		checkpoint_dict = {
			'arch': arch,
			'class_to_idx': model.class_to_idx, 
			'state_dict': model.state_dict(),
			'hidden_units': hidden_units
		}
        
		torch.save(checkpoint_dict, checkpoint)
    
    # Return the model
	return model

data_dir = args.data_dir
train_model(data_dir=args.data_dir, arch=args.arch, hidden_units=args.hidden_units, epochs=args.epochs, learning_rate=args.learning_rate, gpu=args.gpu, checkpoint=args.checkpoint)