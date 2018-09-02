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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

from train import load_model

import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image to predict')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, help='Return top K predictions')
parser.add_argument('--labels', type=str, help='JSON file containing label names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']
    num_labels = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']

    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']
    
# get index to class mapping
loaded_model, class_to_idx = load_checkpoint(args.checkpoint)
idx_to_class = { v : k for k,v in class_to_idx.items()}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage
	
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.set_title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
	
def predict(image_path, model, topk=5, gup=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
	
	# Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    result = model(image).topk(topk)
    
	# Use gpu if selected and available
	if gpu and torch.cuda.is_available():
        top_probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        top_idx = result[1].data.cpu().numpy()[0]
    else:       
        top_probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        top_idx = result[1].data.numpy()[0]
		
    top_classes = [idx_to_class[x] for x in top_idx]
    
    return top_probs, top_classes
	
probs, classes = predict(args.image, args.checkpoint, args.topk)
print(probs)
print(classes)

# TODO: Display an image along with the top 5 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = img.split('/')[-2]
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(8,10), ncols=1, nrows=2)
    flower_name = mapper[img_filename]
    
    ax1.set_title(flower_name)
    ax1.imshow(img)
    ax1.axis('off')
    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()

with open(args.labels, 'r') as f:
	cat_to_name = json.load(f)
	
prob, cls = predict(args.image, args.checkpoint, args.topk)
view_classify(args.image, prob, cls, cat_to_name)