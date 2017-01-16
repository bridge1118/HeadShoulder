import os
import numpy as np
import dicom
import cv2
import matplotlib.pyplot as plt

from natsort import natsorted


class MyImages:
    
    TRAINING_PHASE = 1
    TESTING_PHASE = 0
    
    def __init__(self,im_dir,batch_size,phase=TRAINING_PHASE):
        self.build(im_dir,batch_size,phase)
        
    def build(self,im_dir,batch_size,phase):
        self.phase = phase
        
        if os.path.exists( os.path.join(im_dir,'train') ):
            self.train_urls = os.path.join(im_dir,'train')
        else:
            self.train_urls = im_dir
        
        self.train_imgs = self.readims(self.train_urls)
            
        if (self.phase==1):
            if not os.path.exists( os.path.join(im_dir,'label') ):
                raise ValueError("Error! There is no label images!")
            self.label_urls = os.path.join(im_dir,'label')
            self.label_imgs = self.readims(self.label_urls)
            self.label_imgs = self.label_parsing(self.label_imgs)

        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        
        self.len = self.train_imgs.shape[0]
        
    def readims(self,path):
        
        file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
        
        images = []
        for filename in file_list:
            # check extension dcm/png ?
            ext = filename.split('.')
            if ( ext[ext.__len__()-1]=='png' or ext[ext.__len__()-1]=='jpg' ):
                img = cv2.imread(os.path.join(path,filename),cv2.IMREAD_GRAYSCALE)
            elif ( ext[ext.__len__()-1]=='dcm' ):
                img = dicom.read_file(os.path.join(path,filename))
                img = self.dcmPreprocessing( img.pixel_array )
            if img is not None:
                images.append(img)
        images = np.array(images)
        
        if images.ndim == 3:
            images=np.reshape(images, images.shape + (1,))
        #images = images.astype('float32')
        
        return images
    
    def dcmPreprocessing(self, dcm):
        lower = -600
        grayscale = 255
        
        dcm = dcm.astype('float32')
        
        mx = np.max(dcm[:])
        dcm[ np.where( dcm < lower ) ] = np.nan
        dcm = dcm - lower
        dcm = np.divide( dcm, mx )
        dcm = dcm * grayscale
        dcm[ np.where( np.isnan(dcm) ) ] = 0

        return dcm            
        
    def label_parsing(self,images):
        images[images>0] = 1
        return images
        
    def nextBatch(self, phase=TRAINING_PHASE):
        size = self.train_imgs.shape[0]
        
        if self.end > self.start:
            batch_images = self.train_imgs[self.start:self.end,:,:,:]
            if (self.phase==1):
                batch_labels = self.label_imgs[self.start:self.end,:,:,:]

        else:
            tmp1 = self.train_imgs[self.start:size,:,:,:]
            tmp2 = self.train_imgs[0:self.end,:,:,:]
            batch_images = np.concatenate((tmp1,tmp2),axis=0)
            
            if (self.phase==1):
                tmp3 = self.label_imgs[self.start:size,:,:,:]
                tmp4 = self.label_imgs[0:self.end,:,:,:]
                batch_labels = np.concatenate((tmp3,tmp4),axis=0)
            
        while True:
            self.start = (self.start + self.batch_size) % size
            self.end = (self.end + self.batch_size) % size
            if not self.start>= size+1 and not self.end == 0:
                break
        
        if (self.phase==1):
            return batch_images, batch_labels
        else:
            return batch_images


class MyClassImages:
    
    def __init__(self,im_dir,shape,image_obj=None):
        
        # check path
        if os.path.exists( im_dir ):
            self.__imdir__ = os.path.join(im_dir)
        else:
            raise IOError("IOError! There is no folder in given path!")
        
        # reload imgs or just assign
        if ( image_obj is None ):
            im_size = (shape[1],shape[2]) # height, width
            self.__images__ = Images( Read.readims(self.__imdir__, im_size) )
        elif ( type(image_obj).__module__ == np.__name__ ):
            self.__images__ = Images(image_obj)
        else:
            raise ValueError("ValueError! Please check the input value!")
            
        # check batch
        if ( shape[0]>=1 ):
            self.__batch__ = shape[0]
        else:
            raise ValueError("ValueError! Batch size assignment error!")
        
        # iter variables
        self.__current_b1__ = 0
        
        
    def __str__(self):
        return str( self.shape() )
    
    # override '+' op
    def __add__(self, obj):
        if ( self.shape()[1] != obj.shape()[1] or
             self.shape()[2] != obj.shape()[2] or
             self.shape()[3] != obj.shape()[3] ):
            raise ValueError('Different shape between two given Object!')
        
        new_image = np.concatenate( (self.__images__.__imgs__, obj.__images__.__imgs__), axis=0 )
        new_shape = (self.__batch__,self.shape()[1],self.shape()[2],self.shape()[3])
        return MyClassImages( self.__imdir__, new_shape, new_image )
    
    # make it iteratorable
    def __iter__(self):
        #return iter( self.__images__.__imgs__ )
        return self
    
    def __next__(self):
        if not self.__hasNext__():
            raise StopIteration
        
        self.__current_b1__ = self.__current_b1__ + 1
        return self.__images__.__imgs__[self.__current_b1__-1:self.__current_b1__,:,:,:]
    
    def __hasNext__(self):
        if ( self.__current_b1__>=self.len() ):
            return False
        return True
    
    def len(self):
        return self.__images__.__len__()
    
    def shape(self):
        return self.__images__.__shape__()
        
        
class Images:
    
    def __init__(self,imgs=None):
        
        self.__imgs__ = None
        
        if imgs is not None:
            self.build(imgs)
    
    def isEmpty(self):
        if ( self.__imgs__ is None ):
            return True
        return False
    
    def __shape__(self):
        if self.__imgs__ is not None:
            return self.__imgs__.shape
        return None 
    
    def __len__(self):
        if self.__imgs__ is not None:
            return self.__imgs__.shape[0]
        return None 
        
    def build(self, img):
        self.__imgs__ = img
        
class Read:
    
    @staticmethod
    def readims(path, resize_shape=None):
        
        file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
        
        images = []
        for filename in file_list:
            # check extension dcm/png ?
            ext = filename.split('.')
            if ( ext[ext.__len__()-1]=='dcm' ):
                img = dicom.read_file(os.path.join(path,filename))
                img = img.pixel_array
            else:
                img = cv2.imread(os.path.join(path,filename),cv2.IMREAD_GRAYSCALE)

            if img is not None:
                if resize_shape is not None:
                    img = cv2.resize(img, resize_shape) 
                images.append(img)
                
        images = np.array(images)
        
        if images.ndim == 3:
            images=np.reshape(images, images.shape + (1,))
        #images = images.astype('float32')
        
        return images        
        