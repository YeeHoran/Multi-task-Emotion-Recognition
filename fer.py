''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import pandas as pd


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        if self.split == 'Training':
            self.data = pd.read_csv("./data/train-20240507-yh.csv")
            #self.data = pd.read_csv("./data/outputTrain_forVA - yh-2-20240117.csv")
            self.train_data = self.data['pixels']

            self.train_labels_classify = self.data['emotion']
            self.train_labels_regressV = self.data['Valence']
            self.train_labels_regressA = self.data['Arousal']
            self.train_labels_regressD = self.data['Dominance']
            self.train_data = np.asarray(self.train_data)
        elif self.split == 'PublicTest':
            self.data = pd.read_csv("./data/publictest-20240609-2-yh.csv")
            #self.data = pd.read_csv("./data/3 outputPublicTest_forVAD-20240117.csv")
            self.PublicTest_data = self.data['pixels']
            new_index = np.arange(len(self.PublicTest_data))
            self.PublicTest_data = pd.Series(self.PublicTest_data.values, index=new_index)

            self.PublicTest_labels_classify = self.data['emotion']
            self.PublicTest_labels_regressV = self.data['Valence']
            self.PublicTest_labels_regressA = self.data['Arousal']
            self.PublicTest_labels_regressD = self.data['Dominance']

            self.PublicTest_labels_classify = pd.Series(self.PublicTest_labels_classify.values, index=new_index)
            self.PublicTest_labels_regressV = pd.Series(self.PublicTest_labels_regressV.values, index=new_index)
            self.PublicTest_labels_regressA = pd.Series(self.PublicTest_labels_regressA.values, index=new_index)
            self.PublicTest_labels_regressD = pd.Series(self.PublicTest_labels_regressD.values, index=new_index)

            self.PublicTest_data = np.asarray(self.PublicTest_data)
        else:
            self.data = pd.read_csv("./data/privatetest-20240506-yh.csv")
            #self.data = pd.read_csv("./data/3. outputPrivateTest_forVAD - 20240117.csv")
            self.PrivateTest_data = self.data['pixels']
            new_index = np.arange(len(self.PrivateTest_data))
            self.PrivateTest_data = pd.Series(self.PrivateTest_data.values, index=new_index)

            self.PrivateTest_labels_classify = self.data['emotion']
            self.PrivateTest_labels_regressV = self.data['Valence']
            self.PrivateTest_labels_regressA = self.data['Arousal']
            self.PrivateTest_labels_regressD = self.data['Dominance']

            self.PrivateTest_labels_classify = pd.Series(self.PrivateTest_labels_classify.values, index=new_index)
            self.PrivateTest_labels_regressV = pd.Series(self.PrivateTest_labels_regressV.values, index=new_index)
            self.PrivateTest_labels_regressA = pd.Series(self.PrivateTest_labels_regressA.values, index=new_index)
            self.PrivateTest_labels_regressD = pd.Series(self.PrivateTest_labels_regressD.values, index=new_index)

            self.PrivateTest_data = np.asarray(self.PrivateTest_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target_classify, target_regressV, target_regressA, target_regressD = self.train_data[index], self.train_labels_classify[index], self.train_labels_regressV[index],self.train_labels_regressA[index], self.train_labels_regressD[index]
        elif self.split == 'PublicTest':
            img, target_classify, target_regressV, target_regressA, target_regressD = self.PublicTest_data[index], self.PublicTest_labels_classify[index], self.PublicTest_labels_regressV[index], self.PublicTest_labels_regressA[index], self.PublicTest_labels_regressD[index]
        else:
            img, target_classify, target_regressV, target_regressA, target_regressD = self.PrivateTest_data[index], self.PrivateTest_labels_classify[index], self.PrivateTest_labels_regressV[index], self.PrivateTest_labels_regressA[index], self.PrivateTest_labels_regressD[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        string_list = img.split()
        img = [int(num) for num in string_list]
        img = np.array(img)
        img = img.reshape(48, 48)
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.split == 'Training':
            return img, target_classify, target_regressV, target_regressA, target_regressD
        elif self.split == 'PublicTest':
            return img, target_classify, target_regressV, target_regressA, target_regressD
        else:
            return img, target_classify, target_regressV, target_regressA, target_regressD


    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
