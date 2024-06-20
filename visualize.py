"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

from models.resnet_reg2 import ResNet18RegressionTwoOutputs

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

raw_img = io.imread('images/happy49.jpg')
gray = rgb2gray(raw_img)
gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
img = gray[:, :, np.newaxis]
img = np.concatenate((img, img, img), axis=2)
img = Image.fromarray(img)
inputs = transform_test(img)

###########1. Emotion Category Prediction ####################
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
net = VGG('VGG19')
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

ncrops, c, h, w = np.shape(inputs)
inputs = inputs.view(-1, c, h, w)
inputs = inputs.cuda()
inputs = Variable(inputs, volatile=True)
outputs = net(inputs)
outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

score = F.softmax(outputs_avg)
_, predicted = torch.max(outputs_avg.data, 0)
emotionCategory=str(class_names[int(predicted.cpu().numpy())])
print(emotionCategory) # emotion Category
###########1. Emotion Category Prediction: End ####################


###############2. Valence Prediction###############
net_V = ResNet18RegressionTwoOutputs()
checkpoint_V=torch.load(os.path.join('FER2013_ResNet18RegressionTwoOutputs', 'PrivateTest_model_privateV.t7'))
net_V.load_state_dict(checkpoint_V['net'])
net_V.cuda()
net_V.eval()

outputs = net_V(inputs)
outputs_avg_V = outputs.view(ncrops, -1).mean(0)  # avg over crops
V=outputs_avg_V.item()
print("V:",V)   # V value
###############2. Valence Prediction: End ###############

###############3. Arousal Prediction###############
net_A = ResNet18RegressionTwoOutputs()
checkpoint_A=torch.load(os.path.join('FER2013_ResNet18RegressionTwoOutputs', 'PrivateTest_model_privateA.t7'))
net_A.load_state_dict(checkpoint_A['net'])
net_A.cuda()
net_A.eval()

outputs = net_A(inputs)
outputs_avg_A = outputs.view(ncrops, -1).mean(0)  # avg over crops
A=outputs_avg_A.item()
print("A:",A)
###############2. Valence Prediction: End ###############

###############3. Dominance Prediction###############
net_D = ResNet18RegressionTwoOutputs()
checkpoint_D=torch.load(os.path.join('FER2013_ResNet18RegressionTwoOutputs', 'PublicTest_model_regressD.t7'))
net_D.load_state_dict(checkpoint_D['net'])
net_D.cuda()
net_D.eval()

outputs = net_D(inputs)
outputs_avg_D = outputs.view(ncrops, -1).mean(0)  # avg over crops
D=outputs_avg_D.item()
print("D:",D)
###############2. Valence Prediction: End ###############