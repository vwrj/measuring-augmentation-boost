import torch, torchvision, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, \
        RandomApply, ColorJitter, Grayscale, ToTensor, Lambda
from PIL import Image
from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from einops import rearrange


def improved_jigsaw(img, size=2):
    # create patches
    chunk = img.shape[1] // size
    img = img.unfold(1, chunk, chunk).unfold(2, chunk, chunk)
    img = rearrange(img, 'c x1 x2 h w -> c (x1 x2) h w')
    
    # jigsaw permutation
    perm = torch.randperm(size ** 2)
    img = img[:, perm, :, :]
    img = rearrange(img, 'c (b1 b2) h w -> c (b1 h) (b2 w)', b1=size)
    return img


# Assuming channel first
# Assumptions - image is square, num_pieces is a perfect square
def jigsaw(img, num_pieces=9):
    print(type(img))
    if math.isqrt(num_pieces) ** 2 != num_pieces:
        print("Please use a perfect square")
    elif img.shape[-2] != img.shape[-1]:
        print("Please use a square image")
    elif img.shape[-2] % math.isqrt(num_pieces) != 0:
        print("Make sure the image size and number of pieces are compatible")
    print("Image and num_pieces are compatible")
    # Pixel size of one patch
    piece_size = img.shape[1] // math.isqrt(num_pieces)
    print(piece_size)
    # List to store patches
    pieces = []
    # Extract patches - is there a faster way to do this?
    for i in range(math.isqrt(num_pieces)):
        for j in range(math.isqrt(num_pieces)):
            pieces.append(img[:, piece_size*i:piece_size*(i+1), piece_size*j:piece_size*(j+1)])
    # Shuffle
    np.random.shuffle(pieces)
    # Reassemble
    new_img = np.zeros(img.shape)
    idx = 0
    for i in range(math.isqrt(num_pieces)):
        for j in range(math.isqrt(num_pieces)):
            new_img[:, piece_size*i:piece_size*(i+1), piece_size*j:piece_size*(j+1)] = pieces[idx]
            idx += 1
    return torch.Tensor(new_img)
    
def k2num(s):
    return int(s.split('k')[0]) * 1000

def id2class(class_idx):
    d = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return d[class_idx] + '/'

class CIFAR10DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.root_dir = '/scratch/vr1059/cifar10/data/'

    def train_dataloader(self, augment, size, bs, num_workers=5):
        '''
        augment: boolean flag.
        size: string, such as '1k' or '2k'
        bs: int, denotes batch size.
        
        '''
        if not augment:
            transform = Compose([
                ToTensor(),
            ])
        else:
            transform = Compose([
                RandomRotation(25),
                RandomHorizontalFlip(),
                RandomApply([
                    ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.4, hue = (-0.5, 0.5)),
                    Grayscale(3),
                ]),
                ToTensor(),
                RandomApply([
                    Lambda(lambda x: improved_jigsaw(x)),
                ]), 
            ]) 

        ds = CIFAR10Dataset(self.root_dir, 'train', size, transform=transform) 
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers) 

        return dl

    def val_dataloader(self, size, bs, num_workers=5):
        transform = Compose([
            ToTensor(),
        ])

        ds = CIFAR10Dataset(self.root_dir, 'val', size, transform=transform) 
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers) 

        return dl


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

