

import numpy as np
import torch 
import torchvision
import torchvision.transforms as transforms

class Create():
    """ Creates an permutation matrix,
        a function to permute images,
        and functions to return permuted and original images of MNSIT, FashionMNIST and CIFAR10 datasets
    """
    
    def __init__(self,rows,columns,weighted=False):
        """ rows and columns should be equal to each other and equal to image size as well
            if weighted is True, then permutation matrix will have random numbers between 0 and 1 instead of 1s
        """
        
        self.rows = rows
        self.columns = columns
        if rows!=columns:
            raise(f"Number of Rows {self.rows} not equal to Number of columns {self.columns}")
            return
        
        if weighted == True:
            self.weights = np.random.rand(self.rows)
        else:
            self.weights = np.ones(self.rows)
        # create a list of numbers from zero to the number of row
        self.numbers = np.arange(0,self.rows,1)
        
        # randomly shuffle the above created list. this will serve as index for 1's in permutation matrix
        self.permuted_numbers = np.random.permutation(self.numbers)
        
        # empty matrix of same size as images
        self.out = np.zeros((self.rows,self.columns))
        
        for i in range(self.rows):
            # put 1 in each row and column (index from the list).
            self.out[i,self.permuted_numbers[i]] = self.weights[i]
        
    def matrix(self):
        """ Returns the permutation matrix """
        return self.out
    
    def inverse(self):
        """ Returns the inverse of permutation matrix """
        return np.linalg.inv(self.out)
    
    def permute(self,images):
        """ Use this function to permute your own images
            Images should be a 4d numpy array with shape (N,W,H,Ch)
        """
        
        if len(images.shape) != 4:
            raise("Images should be a 4d numpy array")
            return
        
        output = np.zeros(images.shape)
        
        #permute each image
        for i in range(images.shape[0]):
            #permute each channel, each channel is permuted the same way.
            for j in range(images.shape[3]):
                output[i,:,:,j] = self.out @ images[i,:,:,j] @ self.out
        
        return output
    
    def MNIST(self):
        """ Returns The Original MNIST as well Permuted MNIST images. Permuted image is on same index as its original image
            Also returns the labels for train and test sets.
            The return images and labels are in this order:
            0. Origianl train Images
            1. Original test Images
            2. Permuted train images
            3. permuted test images
            4. labels for training images
            5. labels for test images            
        """
        # load the dataset and create empty tensor to store permuted MNIST
        trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=torchvision.transforms.ToTensor())
        original_images_train = trainset.data
        original_images_test = testset.data
        permuted_images_train = torch.zeros_like(original_images_train)
        permuted_images_test = torch.zeros_like(original_images_test)
        pmt = torch.tensor(self.out).float()
        
        for i in range(permuted_images_train.shape[0]):
            permuted_images_train[i] = pmt @ (original_images_train[i].float()@pmt)
        for i in range(permuted_images_test.shape[0]):
            permuted_images_test[i] = pmt @ (original_images_test[i].float()@pmt)
        
        return original_images_train,original_images_test,permuted_images_train,permuted_images_test,trainset.targets,testset.targets
    
    def FashionMNIST(self):
        """ Returns The Original FashionMNIST as well Permuted Fashion MNIST images. Permuted image is on same index as its original image
            Also returns the labels for train and test sets.
            The return images and labels are in this order:
            0. Origianl train Images
            1. Original test Images
            2. Permuted train images
            3. permuted test images
            4. labels for training images
            5. labels for test images            
        """        
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=torchvision.transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=torchvision.transforms.ToTensor())
        original_images_train = trainset.data
        original_images_test = testset.data
        permuted_images_train = torch.zeros_like(original_images_train)
        permuted_images_test = torch.zeros_like(original_images_test)
        pmt = torch.tensor(self.out).float()
        
        for i in range(permuted_images_test.shape[0]):
            permuted_images_train[i] = pmt @ (original_images_train[i].float()@pmt)
        for i in range(permuted_images_test.shape[0]):
            permuted_images_test[i] = pmt @ (original_images_test[i].float()@pmt)
            
        return original_images_train,original_images_test,permuted_images_train,permuted_images_test,trainset.targets,testset.targets
    
    def CIFAR10(self,grayscale=True):
        """ 
            Returns The Original CIFAR10 as well Permuted CIFAR10 images. Permuted image is on same index as its original image
            Also returns the labels for train and test sets.
            if grayscale is true images will be 1 channel otherwise 3 channel
            The return images and labels are in this order:
            0. Origianl train Images
            1. Original test Images
            2. Permuted train images
            3. permuted test images
            4. labels for training images
            5. labels for test images
            
        """ 
        if grayscale==True:
            # create transformation to convert to tensor and grayscale image and resize to given size.
            tr = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((self.rows,self.columns)),
                torchvision.transforms.ToTensor(),])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=tr)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=tr)
            original_images_train = torch.zeros((50000,self.rows,self.columns))
            original_images_test = torch.zeros((10000,self.rows,self.columns))
            permuted_images_train = torch.zeros((50000,self.rows,self.columns))
            permuted_images_test = torch.zeros((10000,self.rows,self.columns))
            pmt = torch.tensor(self.out).float()
            for i in range(50000):
                original_images_train[i] = tr(trainset.data[i])
                permuted_images_train[i] = pmt @ (original_images_train[i].float()@pmt)
            for i in range(10000):
                original_images_test[i] = tr(testset.data[i])
                permuted_images_test[i] = pmt @ (original_images_test[i].float()@pmt)

            return original_images_train,original_images_test,permuted_images_train,permuted_images_test,trainset.targets,testset.targets            
        else :
            
            tr = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((self.rows,self.columns)),
                torchvision.transforms.ToTensor(),])
            
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=tr)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=tr)
            
            original_images_train = torch.zeros((50000,3,self.rows,self.columns))
            original_images_test = torch.zeros((10000,3,self.rows,self.columns))
            permuted_images_train = torch.zeros((50000,3,self.rows,self.columns))
            permuted_images_test = torch.zeros((10000,3,self.rows,self.columns))
            
            pmt = torch.tensor(self.out).float()

            for i in range(50000):
                for j in range(3):
                    original_images_train[i,j] = tr(trainset.data[i,:,:,j].T)
                    permuted_images_train[i,j] = pmt @ (original_images_train[i,j].float()@pmt)
                    
            for i in range(10000):
                for j in range(3):
                    original_images_test[i,j] = tr(testset.data[i,:,:,j].T)
                    permuted_images_test[i,j] = pmt @ (original_images_test[i,j].float()@pmt)
                    

            return original_images_train,original_images_test,permuted_images_train,permuted_images_test,trainset.targets,testset.targets               
