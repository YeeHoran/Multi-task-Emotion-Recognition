'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.optim as optim
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from models import *

import utils2
import pandas as pd

from models.resnet_reg2 import ResNet18RegressionTwoOutputs

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model_classify', type=str, default='ResNet18', help='CNN architecture')
parser.add_argument('--model_regressV', type=str, default='ResNet18RegressionTwoOutputs', help='CNN architecture')
parser.add_argument('--model_regressA', type=str, default='ResNet18RegressionTwoOutputs', help='CNN architecture')
parser.add_argument('--model_regressD', type=str, default='ResNet18RegressionTwoOutputs', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
# parser.add_argument('--dataset', type=str, default='CK+', help='CNN architecture')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

# for classify
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
# for V
best_PublicTest_AveragelossV = torch.tensor(float('inf'), dtype=torch.float32)  # best PublicTest accuracy
best_PublicTest_epoch_lossV = 0
best_PrivateTest_epoch_lossV = 0
best_PrivateTest_AveragelossV = torch.tensor(float('inf'), dtype=torch.float32)  # best PrivateTest accuracy
# for A
best_PublicTest_AveragelossA = torch.tensor(float('inf'), dtype=torch.float32)  # best PublicTest accuracy
best_PublicTest_epoch_lossA = 0
best_PrivateTest_epoch_lossA = 0
best_PrivateTest_AveragelossA = torch.tensor(float('inf'), dtype=torch.float32)  # best PrivateTest accuracy
# for D
best_PublicTest_AveragelossD = torch.tensor(float('inf'), dtype=torch.float32)  # best PublicTest accuracy
best_PublicTest_epoch_lossD = 0
best_PrivateTest_epoch_lossD = 0
best_PrivateTest_AveragelossD = torch.tensor(float('inf'), dtype=torch.float32)  # best PrivateTest accuracy

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.9  # 0.9

cut_size = 44
# total_epoch = 2
total_epoch = 120

path_classify = os.path.join(opt.dataset + '_' + opt.model_classify)
path_regressV = os.path.join(opt.dataset + '_' + opt.model_regressV)
path_regressA = os.path.join(opt.dataset + '_' + opt.model_regressA)
path_regressD = os.path.join(opt.dataset + '_' + opt.model_regressD)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def VADLabelFileRead():  # read in the VAD and label data in the excel.
    # Replace 'file.xlsx' with the path to your Excel file
    excel_file = pd.ExcelFile("./data/VAD_category_consistency_loss_all-20240220.xlsx")
    sheet_names = excel_file.sheet_names
    dfs = []
    # Read each sheet into a separate DataFrame and append it to the list
    for sheet in sheet_names:
        df = pd.read_excel(excel_file, sheet)
        dfs.append(df)
    # Concatenate the DataFrames along a new axis to create a 3D ndarray
    VADLabel = np.concatenate([df.values[np.newaxis, :, :] for df in dfs], axis=0)
    return VADLabel


def consistency(label, V, A, D):  # according to the consistency between category and V to calculate the loss.
    consistlossSum = 0.0
    consistloss = 0.0

    length = len(label)
    tensor_cpu_label = label.cpu()
    ndarray_label = tensor_cpu_label.numpy()
    label = ndarray_label
    tensor_cpu_V = V.cpu()
    ndarray_V = tensor_cpu_V.numpy()
    V = ndarray_V
    tensor_cpu_A = A.cpu()
    ndarray_A = tensor_cpu_A.numpy()
    A = ndarray_A
    tensor_cpu_D = D.cpu()
    ndarray_D = tensor_cpu_D.numpy()
    D = ndarray_D

    Label_VAD_ThanZero=VADLabelFileRead()

    for index in range(length):  # the indexth img in the batch
        print("index" + str(index))
        #for dimension in range(3):   # range(3): represent 'V', 'A', 'D' respectively.

        # V's,  2nd dimension: 0-2, V is 0; 1st dimension is label
        if V[index] > 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][0][1]
        elif V[index] == 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][0][2]
        else:
            consistlossSum += Label_VAD_ThanZero[label[index]][0][3]

        # A's, 2nd dimension: 0-2, A is 1; 1st dimension is label
        if A[index] > 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][1][1]
        elif A[index] == 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][1][2]
        else:
            consistlossSum += Label_VAD_ThanZero[label[index]][1][3]

        # D's , 2nd dimension: 0-2, D is 2; 1st dimension is label
        if D[index] > 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][2][1]
        elif D[index] == 0.0:
            consistlossSum += Label_VAD_ThanZero[label[index]][2][2]
        else:
            consistlossSum += Label_VAD_ThanZero[label[index]][2][3]

    return consistlossSum  # 返回一致性约束生成的loss计算结果。


def custom_transform(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    custom_transform,
])

trainset = FER2013(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split='PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

net_classify = ResNet18()
net_regressV = ResNet18RegressionTwoOutputs()
net_regressA = ResNet18RegressionTwoOutputs()
net_regressD = ResNet18RegressionTwoOutputs()

#################################################################
'''
if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')
'''
####################################################################

if use_cuda:
    net_classify.cuda()
    net_regressV.cuda()
    net_regressA.cuda()
    net_regressD.cuda()

criterion_classify = nn.CrossEntropyLoss()
criterion_regress = nn.MSELoss()
optimizer = optim.SGD(list(net_classify.parameters()) + list(net_regressV.parameters()) + list(net_regressA.parameters()) + list(net_regressD.parameters()), lr=opt.lr, momentum=0.9,
                      weight_decay=5e-4)

# create 3 list to save the accuracies of train, public test, and private test.
trainAccuracyList_classify = list()
pubtestAccuracyList_classify = list()
privatetestAccuracyList_classify = list()

trainLossList_regressV = list()
pubtestLossList_regressV = list()
privatetestLossList_regressV = list()
trainLossList_regressA = list()
pubtestLossList_regressA = list()
privatetestLossList_regressA = list()
trainLossList_regressD = list()
pubtestLossList_regressD = list()
privatetestLossList_regressD = list()

Train_acc_classify = 0.0
Train_loss_regressV = 0.0
Train_loss_regressA = 0.0
Train_loss_regressD = 0.0


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc_classify

    total_loss_regressV = 0.0
    total_samplesV = 0
    total_loss_regressA = 0.0
    total_samplesA = 0
    total_loss_regressD = 0.0
    total_samplesD = 0

    net_classify.train()
    net_regressV.train()
    net_regressA.train()
    net_regressD.train()

    train_loss_classify = 0.0
    train_loss_regressV = 0.0
    train_loss_regressA = 0.0
    train_loss_regressD = 0.0
    correct_classify = 0
    total_regressV = 0
    total_regressA = 0
    total_regressD = 0
    total_classify = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, target_classify, target_regressV, target_regressA, target_regressD) in enumerate(trainloader):
        inputs, target_classify, target_regressV, target_regressA, target_regressD = inputs.float(), target_classify, target_regressV.float(), target_regressA.float(), target_regressD.float()
        if use_cuda:
            inputs, target_classify, target_regressV, target_regressA, target_regressD = inputs.cuda(), target_classify.cuda(), target_regressV.cuda(), target_regressA.cuda(), target_regressD.cuda()
        optimizer.zero_grad()
        inputs, target_classify, target_regressV, target_regressA, target_regressD = Variable(inputs), Variable(target_classify), Variable(
            target_regressV), Variable(target_regressA), Variable(target_regressD)


        # forward pass for classify
        outputs_classify = net_classify(inputs)
        train_loss_classify = criterion_classify(outputs_classify, target_classify)
        diff = utils2.orth_dist(net_classify.layer2[0].shortcut[0].weight) + utils2.orth_dist(
            net_classify.layer3[0].shortcut[0].weight) + utils2.orth_dist(net_classify.layer4[0].shortcut[0].weight)
        diff += utils2.deconv_orth_dist(net_classify.layer1[0].conv1.weight, stride=1) + utils2.deconv_orth_dist(
            net_classify.layer1[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_classify.layer2[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_classify.layer2[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_classify.layer3[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_classify.layer3[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_classify.layer4[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_classify.layer4[1].conv1.weight, stride=1)
        train_loss_classify += diff * 0.5

        # forward pass for regressionV A D respectively
        outputs_regressV = net_regressV(inputs)
        outputs_regressA = net_regressA(inputs)
        outputs_regressD = net_regressD(inputs)
        train_loss_regressV = criterion_regress(outputs_regressV, target_regressV)
        train_loss_regressA = criterion_regress(outputs_regressA, target_regressA)
        train_loss_regressD = criterion_regress(outputs_regressD, target_regressD)
        # for V first's orth_loss:
        diff = utils2.orth_dist(net_regressV.layer2[0].shortcut[0].weight) + utils2.orth_dist(
            net_regressV.layer3[0].shortcut[0].weight) + utils2.orth_dist(net_regressV.layer4[0].shortcut[0].weight)
        diff += utils2.deconv_orth_dist(net_regressV.layer1[0].conv1.weight, stride=1) + utils2.deconv_orth_dist(
            net_regressV.layer1[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressV.layer2[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressV.layer2[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressV.layer3[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressV.layer3[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressV.layer4[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressV.layer4[1].conv1.weight, stride=1)
        train_loss_regressV += diff * 0.5
        # 求class_predict和loss_regressV两个network的loss总和
        total_loss = train_loss_classify + train_loss_regressV  # 这是两个network的正常loss之和

        # for A secondly:
        diff = utils2.orth_dist(net_regressA.layer2[0].shortcut[0].weight) + utils2.orth_dist(
            net_regressA.layer3[0].shortcut[0].weight) + utils2.orth_dist(net_regressA.layer4[0].shortcut[0].weight)
        diff += utils2.deconv_orth_dist(net_regressA.layer1[0].conv1.weight, stride=1) + utils2.deconv_orth_dist(
            net_regressA.layer1[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressA.layer2[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressA.layer2[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressA.layer3[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressA.layer3[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressA.layer4[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressA.layer4[1].conv1.weight, stride=1)
        train_loss_regressA += diff * 0.5

        # 求两个network的loss总和
        total_loss = total_loss + train_loss_regressA  # 这是在现有loss加上train_loss_regressA

        # for D thirdly:
        diff = utils2.orth_dist(net_regressD.layer2[0].shortcut[0].weight) + utils2.orth_dist(
            net_regressD.layer3[0].shortcut[0].weight) + utils2.orth_dist(net_regressD.layer4[0].shortcut[0].weight)
        diff += utils2.deconv_orth_dist(net_regressD.layer1[0].conv1.weight, stride=1) + utils2.deconv_orth_dist(
            net_regressD.layer1[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressD.layer2[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressD.layer2[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressD.layer3[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressD.layer3[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net_regressD.layer4[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net_regressD.layer4[1].conv1.weight, stride=1)
        train_loss_regressD += diff * 0.5

        # 求两个network的loss总和
        total_loss = total_loss + train_loss_regressD  # 这是在现有loss加

        # 求分类和回归的一致性约束loss, to here.上train_loss_regressD，目前是classify, V, A, D的预测loss之sum.
        _, predicted_classify = torch.max(outputs_classify.data, 1)
        label_tensors_classify = torch.tensor(predicted_classify, dtype=torch.int32)

        label_tensors_regressV = torch.tensor(outputs_regressV, dtype=torch.float32)
        label_tensors_regressV = torch.flatten(label_tensors_regressV)

        label_tensors_regressA = torch.tensor(outputs_regressA, dtype=torch.float32)
        label_tensors_regressA = torch.flatten(label_tensors_regressA)

        label_tensors_regressD = torch.tensor(outputs_regressD, dtype=torch.float32)
        label_tensors_regressD = torch.flatten(label_tensors_regressD)

        consist_loss = consistency(label_tensors_classify, label_tensors_regressV, label_tensors_regressA, label_tensors_regressD)    #this is the most important, get consist for a batch of imgs among according to their label, V, A and D.
        total_loss += consist_loss  # 求总loss = 4 network loss+consistency loss(label, VAD consistency for the batch of imgs)

        total_loss.backward()  ############# to here, backward problem.solved, to float before entering criterion. it does this just after load data from batch in this case.
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        # 求分类的正确率
        total_classify += target_classify.size(0)
        correct_classify += predicted_classify.eq(target_classify.data).cpu().sum()
        '''
        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''

        # 求回归V值的loss
        total_loss_regressV = train_loss_regressV + train_loss_regressV.item()
        total_samplesV = total_samplesV + target_regressV.size(0)
        # 求回归A值的loss
        total_loss_regressA = total_loss_regressA + train_loss_regressA.item()
        total_samplesA = total_samplesA + target_regressA.size(0)
        # 求回归D值的loss
        total_loss_regressD = total_loss_regressD + train_loss_regressD.item()
        total_samplesD = total_samplesD + target_regressD.size(0)
        '''utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (total_loss / (batch_idx + 1), 100. * correct / total, correct, total))'''
    average_lossV = total_loss_regressV / total_samplesV
    average_lossA = total_loss_regressA / total_samplesA
    average_lossD = total_loss_regressD / total_samplesD
    print(f'Average Train LossV: {average_lossV:.3f} ')
    trainLossList_regressV.append(average_lossV)
    print(f'Average Train LossA: {average_lossA:.3f} ')
    trainLossList_regressA.append(average_lossA)
    print(f'Average Train LossD: {average_lossD:.3f} ')
    trainLossList_regressD.append(average_lossD)

    train_acc_classify = 100. * correct_classify / total_classify
    trainAccuracyList_classify.append(train_acc_classify)


def PublicTest(epoch):
    net_classify.eval()
    net_regressV.eval()
    net_regressA.eval()
    net_regressD.eval()

    PublicTest_loss_classify = 0.0
    PublicTest_loss_regressV = 0.0
    PublicTest_loss_regressA = 0.0
    PublicTest_loss_regressD = 0.0

    correct_classify = 0
    total_classify = 0

    total_regressV = 0
    total_regressA = 0
    total_regressD = 0

    for batch_idx, (inputs, target_classify, target_regressV, target_regressA, target_regressD) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, target_classify, target_regressV, target_regressA, target_regressD = inputs.cuda(), target_classify.cuda(), target_regressV.cuda(), target_regressA.cuda(), target_regressD.cuda()

        with torch.no_grad():
            inputs, target_classify, target_regressV, target_regressA, target_regressD = Variable(inputs), Variable(target_classify), Variable(
                target_regressV), Variable(target_regressA), Variable(target_regressD)
        ############
        # forward pass for classify&regress
        outputs_classify = net_classify(inputs)
        outputs_regressV = net_regressV(inputs)
        outputs_regressA = net_regressA(inputs)
        outputs_regressD = net_regressD(inputs)

        outputs_avg_classify = outputs_classify.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_classify = criterion_classify(outputs_avg_classify, target_classify)
        PublicTest_loss_classify += loss_classify.data

        _, predicted_classify = torch.max(outputs_avg_classify.data, 1)
        total_classify += target_classify.size(0)
        correct_classify += predicted_classify.eq(target_classify.data).cpu().sum()  # solved.
        '''
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''
        #for V regress loss
        outputs_avg_regressV = outputs_regressV.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressV = criterion_regress(outputs_avg_regressV, target_regressV)
        PublicTest_loss_regressV += loss_regressV.data
        total_regressV+= target_regressV.size(0)  # total_regress: total number of samples

        # for A regress loss
        outputs_avg_regressA = outputs_regressA.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressA = criterion_regress(outputs_avg_regressA, target_regressA)
        PublicTest_loss_regressA += loss_regressA.data
        total_regressA += target_regressA.size(0)  # total_regress: total number of samples

        # for D regress loss
        outputs_avg_regressD = outputs_regressD.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressD = criterion_regress(outputs_avg_regressD, target_regressD)
        PublicTest_loss_regressD += loss_regressD.data
        total_regressD += target_regressD.size(0)  # total_regress: total number of samples


    # Save checkpoint: classify.
    PublicTest_acc = 100. * correct_classify / total_classify
    pubtestAccuracyList_classify.append(PublicTest_acc)
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net_classify.state_dict() if use_cuda else net_classify,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path_classify):
            os.mkdir(path_classify)
        torch.save(state, os.path.join(path_classify, 'PublicTest_model_classify.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

    # save checkpoint: V regress
    PublicTest_av_lossV = PublicTest_loss_regressV / total_regressV
    PublicTest_av_lossV = PublicTest_av_lossV.item()
    pubtestLossList_regressV.append(PublicTest_av_lossV)
    print(f'PublicTest_av_lossV: {PublicTest_av_lossV:.3f} ')
    global best_PublicTest_AveragelossV  # best PublicTest accuracy
    global best_PublicTest_epoch_lossV
    if PublicTest_av_lossV < best_PublicTest_AveragelossV:
        best_PublicTest_AveragelossV = PublicTest_av_lossV
        best_PublicTest_epoch_lossV = epoch
        print('SavingV..')
        print("best_PublicTest_AveragelossV: %0.3f" % PublicTest_av_lossV)
        state = {
            'net': net_regressV.state_dict() if use_cuda else net_regressV,
            'loss': PublicTest_av_lossV,
            'epoch': epoch,
        }
        if not os.path.isdir(path_regressV):
            os.mkdir(path_regressV)
        torch.save(state, os.path.join(path_regressV, 'PublicTest_model_regressV.t7'))

    # save checkpoint: A regress
    PublicTest_av_lossA = PublicTest_loss_regressA / total_regressA
    PublicTest_av_lossA = PublicTest_av_lossA.item()
    pubtestLossList_regressA.append(PublicTest_av_lossA)
    print(f'PublicTest_av_lossA: {PublicTest_av_lossA:.3f} ')
    global best_PublicTest_AveragelossA  # best PublicTest accuracy
    global best_PublicTest_epoch_lossA
    if PublicTest_av_lossA < best_PublicTest_AveragelossA:
        best_PublicTest_AveragelossA = PublicTest_av_lossA
        best_PublicTest_epoch_lossA = epoch
        print('SavingA..')
        print("best_PublicTest_AveragelossA: %0.3f" % PublicTest_av_lossA)
        state = {
            'net': net_regressA.state_dict() if use_cuda else net_regressA,
            'loss': PublicTest_av_lossA,
            'epoch': epoch,
        }
        if not os.path.isdir(path_regressA):
            os.mkdir(path_regressA)
        torch.save(state, os.path.join(path_regressA, 'PublicTest_model_regressA.t7'))

    # save checkpoint: D regress
    PublicTest_av_lossD = PublicTest_loss_regressD / total_regressD
    PublicTest_av_lossD = PublicTest_av_lossD.item()
    pubtestLossList_regressD.append(PublicTest_av_lossD)
    print(f'PublicTest_av_lossD: {PublicTest_av_lossD:.3f} ')
    global best_PublicTest_AveragelossD  # best PublicTest accuracy
    global best_PublicTest_epoch_lossD
    if PublicTest_av_lossD < best_PublicTest_AveragelossD:
        best_PublicTest_AveragelossD = PublicTest_av_lossD
        best_PublicTest_epoch_lossD = epoch
        print('SavingD..')
        print("best_PublicTest_AveragelossD: %0.3f" % PublicTest_av_lossD)
        state = {
            'net': net_regressD.state_dict() if use_cuda else net_regressD,
            'loss': PublicTest_av_lossD,
            'epoch': epoch,
        }
        if not os.path.isdir(path_regressD):
            os.mkdir(path_regressD)
        torch.save(state, os.path.join(path_regressD, 'PublicTest_model_regressD.t7'))


def PrivateTest(epoch):
    net_classify.eval()
    net_regressV.eval()
    net_regressA.eval()
    net_regressD.eval()

    PrivateTest_loss_classify = 0.0
    PrivateTest_loss_regressV = 0.0
    PrivateTest_loss_regressA = 0.0
    PrivateTest_loss_regressD = 0.0

    correct_classify = 0
    total_classify = 0

    total_regressSampleV = 0
    total_regressSampleA = 0
    total_regressSampleD = 0

    for batch_idx, (inputs, target_classify, target_regressV, target_regressA, target_regressD) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, target_classify, target_regressV, target_regressA, target_regressD = inputs.cuda(), target_classify.cuda(), target_regressV.cuda(), target_regressA.cuda(), target_regressD.cuda()
        with torch.no_grad():
            inputs, target_classify, target_regressV, target_regressA, target_regressD = Variable(inputs), Variable(target_classify), Variable(
                target_regressV), Variable(target_regressA), Variable(target_regressD)

        # forward pass for classify&regress
        outputs_classify = net_classify(inputs)
        outputs_regressV = net_regressV(inputs)
        outputs_regressA = net_regressA(inputs)
        outputs_regressD = net_regressD(inputs)

        # classify handle
        outputs_avg_classify = outputs_classify.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_classify = criterion_classify(outputs_avg_classify, target_classify)
        PrivateTest_loss_classify += loss_classify.data

        _, predicted_classify = torch.max(outputs_avg_classify.data, 1)
        total_classify += target_classify.size(0)
        correct_classify += predicted_classify.eq(target_classify.data).cpu().sum()

        '''
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''
        # regress handle
        # for V
        outputs_avg_regressV = outputs_regressV.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressV = criterion_regress(outputs_avg_regressV, target_regressV)
        PrivateTest_loss_regressV += loss_regressV.data
        total_regressSampleV += target_regressV.size(0)

        # for A
        outputs_avg_regressA = outputs_regressA.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressA = criterion_regress(outputs_avg_regressA, target_regressA)
        PrivateTest_loss_regressA += loss_regressA.data
        total_regressSampleA += target_regressA.size(0)

        # for D
        outputs_avg_regressD = outputs_regressD.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss_regressD = criterion_regress(outputs_avg_regressD, target_regressD)
        PrivateTest_loss_regressD += loss_regressD.data
        total_regressSampleD += target_regressD.size(0)

    # Save checkpoint.
    # for classify
    PrivateTest_acc = 100. * correct_classify / total_classify
    privatetestAccuracyList_classify.append(PrivateTest_acc)

    # for classify
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    if PrivateTest_acc > best_PrivateTest_acc:
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net_classify.state_dict() if use_cuda else net_classify,
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path_classify):
            os.mkdir(path_classify)
        torch.save(state, os.path.join(path_classify, 'PrivateTest_model_classify.t7'))

    # for V regress
    PrivateTest_av_lossV = PrivateTest_loss_regressV / total_regressSampleV
    PrivateTest_av_lossV = PrivateTest_av_lossV.item()
    privatetestLossList_regressV.append(PrivateTest_av_lossV)
    for loss in privatetestLossList_regressV:
        print(f'privatetestLossList_regressV: {loss:.3f}')
    global best_PrivateTest_AveragelossV
    global best_PrivateTest_epoch_lossV
    if PrivateTest_av_lossV < best_PrivateTest_AveragelossV:  # 改为<=
        best_PrivateTest_AveragelossV = PrivateTest_av_lossV
        best_PrivateTest_Averageloss_epochV = epoch
        print('SavingV..')
        print("best_PrivateTest_AveragelossV: %0.3f" % best_PrivateTest_AveragelossV)
        state = {
            'net': net_regressV.state_dict() if use_cuda else net_regressV,
            'best_PublicTest_AveragelossV': best_PublicTest_AveragelossV,
            'best_PrivateTest_AveragelossV': best_PrivateTest_AveragelossV,
            'best_PublicTest_acc_epochV': best_PublicTest_epoch_lossV,
            'best_PrivateTest_Averageloss_epochV': best_PrivateTest_Averageloss_epochV,
        }
        if not os.path.isdir(path_regressV):
            os.mkdir(path_regressV)
        torch.save(state, os.path.join(path_regressV, 'PrivateTest_model_privateV.t7'))

    # for A regress
    PrivateTest_av_lossA = PrivateTest_loss_regressA / total_regressSampleA
    PrivateTest_av_lossA = PrivateTest_av_lossA.item()
    privatetestLossList_regressA.append(PrivateTest_av_lossA)
    for loss in privatetestLossList_regressA:
        print(f'privatetestLossList_regressA: {loss:.3f}')
    global best_PrivateTest_AveragelossA
    global best_PrivateTest_epoch_lossA
    if PrivateTest_av_lossA < best_PrivateTest_AveragelossA:  # 改为<=
        best_PrivateTest_AveragelossA = PrivateTest_av_lossA
        best_PrivateTest_Averageloss_epochA = epoch
        print('SavingA..')
        print("best_PrivateTest_AveragelossA: %0.3f" % best_PrivateTest_AveragelossA)
        state = {
            'net': net_regressA.state_dict() if use_cuda else net_regressA,
            'best_PublicTest_AveragelossA': best_PublicTest_AveragelossA,
            'best_PrivateTest_AveragelossA': best_PrivateTest_AveragelossA,
            'best_PublicTest_acc_epochA': best_PublicTest_epoch_lossA,
            'best_PrivateTest_Averageloss_epochA': best_PrivateTest_Averageloss_epochA,
        }
        if not os.path.isdir(path_regressA):
            os.mkdir(path_regressA)
        torch.save(state, os.path.join(path_regressA, 'PrivateTest_model_privateA.t7'))

    # for D regress
    PrivateTest_av_lossD = PrivateTest_loss_regressD / total_regressSampleD
    PrivateTest_av_lossD = PrivateTest_av_lossD.item()
    privatetestLossList_regressD.append(PrivateTest_av_lossD)
    for loss in privatetestLossList_regressD:
        print(f'privatetestLossList_regressD: {loss:.3f}')
    global best_PrivateTest_AveragelossD
    global best_PrivateTest_epoch_lossD
    if PrivateTest_av_lossD < best_PrivateTest_AveragelossD:  # 改为<=
        best_PrivateTest_AveragelossD = PrivateTest_av_lossD
        best_PrivateTest_Averageloss_epochD = epoch
        print('SavingD..')
        print("best_PrivateTest_AveragelossD: %0.3f" % best_PrivateTest_AveragelossD)
        state = {
            'net': net_regressD.state_dict() if use_cuda else net_regressD,
            'best_PublicTest_AveragelossD': best_PublicTest_AveragelossD,
            'best_PrivateTest_AveragelossD': best_PrivateTest_AveragelossD,
            'best_PublicTest_acc_epochD': best_PublicTest_epoch_lossD,
            'best_PrivateTest_Averageloss_epochD': best_PrivateTest_Averageloss_epochD,
        }
        if not os.path.isdir(path_regressD):
            os.mkdir(path_regressD)
        torch.save(state, os.path.join(path_regressD, 'PrivateTest_model_privateA.t7'))


if __name__ == '__main__':
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        PublicTest(epoch)
        PrivateTest(epoch)

        # add by HY
        data = open("data.txt", 'a')
        # classify
        print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc, file=data)
        print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch, file=data)
        print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc, file=data)
        print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch, file=data)
        # regress V
        print("best_PublicTest_AveragelossV: %0.3f" % best_PublicTest_AveragelossV, file=data)
        print("best_PublicTest_epoch_lossV: %d" % best_PublicTest_epoch_lossV, file=data)
        print("best_PrivateTest_AveragelossV: %0.3f" % best_PrivateTest_AveragelossV, file=data)
        print("best_PrivateTest_epoch_lossV: %d" % best_PrivateTest_epoch_lossV, file=data)

        # regress A
        print("best_PublicTest_AveragelossA: %0.3f" % best_PublicTest_AveragelossA, file=data)
        print("best_PublicTest_epoch_lossA: %d" % best_PublicTest_epoch_lossA, file=data)
        print("best_PrivateTest_AveragelossA: %0.3f" % best_PrivateTest_AveragelossA, file=data)
        print("best_PrivateTest_epoch_lossA: %d" % best_PrivateTest_epoch_lossA, file=data)

        # regress D
        print("best_PublicTest_AveragelossD: %0.3f" % best_PublicTest_AveragelossD, file=data)
        print("best_PublicTest_epoch_lossD: %d" % best_PublicTest_epoch_lossD, file=data)
        print("best_PrivateTest_AveragelossD: %0.3f" % best_PrivateTest_AveragelossD, file=data)
        print("best_PrivateTest_epoch_lossD: %d" % best_PrivateTest_epoch_lossD, file=data)

        data.close()
        # add by HY

    # print out the best classify and regress states
    # classify
    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
    # regress V
    print("best_PublicTest_AveragelossV: %0.3f" % best_PublicTest_AveragelossV)
    print("best_PublicTest_epoch_lossV: %d" % best_PublicTest_epoch_lossV)
    print("best_PrivateTest_AveragelossV: %0.3f" % best_PrivateTest_AveragelossV)
    print("best_PrivateTest_epoch_lossV: %d" % best_PrivateTest_epoch_lossV)
    # regress A
    print("best_PublicTest_AveragelossA: %0.3f" % best_PublicTest_AveragelossA)
    print("best_PublicTest_epoch_lossA: %d" % best_PublicTest_epoch_lossA)
    print("best_PrivateTest_AveragelossA: %0.3f" % best_PrivateTest_AveragelossA)
    print("best_PrivateTest_epoch_lossA: %d" % best_PrivateTest_epoch_lossA)
    # regress D
    print("best_PublicTest_AveragelossD: %0.3f" % best_PublicTest_AveragelossD)
    print("best_PublicTest_epoch_lossD: %d" % best_PublicTest_epoch_lossD)
    print("best_PrivateTest_AveragelossD: %0.3f" % best_PrivateTest_AveragelossD)
    print("best_PrivateTest_epoch_lossD: %d" % best_PrivateTest_epoch_lossD)

    # save the final best model's parameters to file as well.
    # classify
    data = open("data.txt", 'a')
    print("best model is:", file=data)
    print("best_PublicTest_Acc: %0.3f" % best_PublicTest_acc, file=data)
    print("best_PublicTest_Acc_epoch: %d" % best_PublicTest_acc_epoch, file=data)
    print("best_PrivateTest_Acc: %0.3f" % best_PrivateTest_acc, file=data)
    print("best_PrivateTest_Acc_epoch: %d" % best_PrivateTest_acc_epoch, file=data)
    # regress V
    print("best_PublicTest_AveragelossV: %0.3f" % best_PublicTest_AveragelossV, file=data)
    print("best_PublicTest_epoch_lossV: %d" % best_PublicTest_epoch_lossV, file=data)
    print("best_PrivateTest_AveragelossV: %0.3f" % best_PrivateTest_AveragelossV, file=data)
    print("best_PrivateTest_epoch_lossV: %d" % best_PrivateTest_epoch_lossV, file=data)

    # regress A
    print("best_PublicTest_AveragelossA: %0.3f" % best_PublicTest_AveragelossA, file=data)
    print("best_PublicTest_epoch_lossA: %d" % best_PublicTest_epoch_lossA, file=data)
    print("best_PrivateTest_AveragelossA: %0.3f" % best_PrivateTest_AveragelossA, file=data)
    print("best_PrivateTest_epoch_lossA: %d" % best_PrivateTest_epoch_lossA, file=data)

    # regress D
    print("best_PublicTest_AveragelossD: %0.3f" % best_PublicTest_AveragelossD, file=data)
    print("best_PublicTest_epoch_lossD: %d" % best_PublicTest_epoch_lossD, file=data)
    print("best_PrivateTest_AveragelossD: %0.3f" % best_PrivateTest_AveragelossD, file=data)
    print("best_PrivateTest_epoch_lossD: %d" % best_PrivateTest_epoch_lossD, file=data)
    data.close()

    # save the classify process of accuracies in each epoch to csv file, including train, publictest, and privatetest.
    # Create a DataFrame
    column_heads = ['TrainAcc', 'PubtestAcc', 'PritestAcc']
    df_classify = pd.DataFrame(
        list(zip(trainAccuracyList_classify, pubtestAccuracyList_classify, privatetestAccuracyList_classify)),
        columns=column_heads)
    # Specify the file path
    csv_file_path = 'AccProcess_classify.csv'
    # Save the DataFrame to a CSV file
    df_classify.to_csv(csv_file_path, index=False)

    # save the regress V process of losses in each epoch to csv file, including trainV, publictestV, and privatetestV.
    column_heads = ['TrainLossV', 'PubtestLossV', 'PritestLossV']
    df_regressV = pd.DataFrame(
        list(zip(trainLossList_regressV, pubtestLossList_regressV, privatetestLossList_regressV)), columns=column_heads)
    csv_file_path = 'AccProcess_regressV.csv'
    df_regressV.to_csv(csv_file_path)

    # save the regress A process of losses in each epoch to csv file, including trainA, publictestA, and privatetestA.
    column_heads = ['TrainLossA', 'PubtestLossA', 'PritestLossA']
    df_regressA = pd.DataFrame(
        list(zip(trainLossList_regressA, pubtestLossList_regressA, privatetestLossList_regressA)), columns=column_heads)
    csv_file_path = 'AccProcess_regressA.csv'
    df_regressA.to_csv(csv_file_path)

    # save the regress D process of losses in each epoch to csv file, including trainD, publictestD, and privatetestD.
    column_heads = ['TrainLossD', 'PubtestLossD', 'PritestLossD']
    df_regressD = pd.DataFrame(
        list(zip(trainLossList_regressD, pubtestLossList_regressD, privatetestLossList_regressD)), columns=column_heads)
    csv_file_path = 'AccProcess_regressD.csv'
    df_regressD.to_csv(csv_file_path)
