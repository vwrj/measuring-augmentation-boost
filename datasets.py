import torch, torchvision, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, \
        RandomApply, ColorJitter, Grayscale, ToTensor, Lambda
from PIL import Image

def k2num(s):
    return int(s.split('k')[0]) * 1000

def id2class(class_idx):
    d = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return d[class_idx] + '/'


class CIFAR10Dataset(Dataset):
    
    def __init__(self, root_dir, phase, size, transform = None):
        '''
        root_dir: location of dataset
        phase: one of {train, val, test}
        size: string with the format "{number}k" like 1k, or 5k. 

        '''
        self.phase = phase
        self.size = k2num(size)
        if self.size > 49000:
            raise ValueError('Training size must not be more than 49k, because validation is set to the last 1k.')
        self.chunk = self.size // 10
        self.dir = root_dir + 'train/' if phase in ['train', 'val'] else root_dir + 'test/'
        self.transform = transform
        
        if self.phase == 'val':
            # Freeze the last 1k images (last 100 per class) as the validation set. 
            # When we train, we will only train up to 49k images from the train set. 
            self.size = k2num('1k')
        elif self.phase == 'test':
            # Set test set to be constant size. 
            self.size = k2num('10k')
          
    def __len__(self):
        return self.size
    
    def id2fn(self, img_id):
        img_id = str(img_id)
        for _ in range(4 - len(img_id)):
            img_id = '0' + img_id
            
        return img_id + '.png'
        
    def __getitem__(self, idx):
        '''
        .png images are stored in self.dir under 10 folders (10 classes). 
        Each folder has 5000 images ranging from '0001.png' to '5000.png'. 
        
        Assume size is 1k, or 1000. This corresponds to 100 images per class. 
        idx is going to be in the range [0, 999], inclusive.
        '''
        
        class_id = idx // self.chunk
        img_id = (idx % self.chunk) + 1
        
        if self.phase == 'val':
            img_id = 5000 - (idx % 100)
        
        img = Image.open(self.dir + id2class(class_id) + self.id2fn(img_id))
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_id, img_id


'''
** 1k case **
900 --> class 9, 1.png
901 --> class 9, 2.png
...
999 --> class 9, 100.png

if val:
    img_id = 5000 - (idx % 100)
    
900 --> class 9, 5000.png
999 --> class 9, 4901.png
'''

