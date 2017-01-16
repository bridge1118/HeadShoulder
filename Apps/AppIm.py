import os
import numpy as np
import dicom
import cv2

from natsort import natsorted

class MyClassImages:
    
    # Members:
    # __imdir__
    # __images__
    # __batch__
    
    def __init__(self,im_dir,shape,image_obj=None,scaling=False):
        
        # check path
        if os.path.exists( im_dir ):
            self.__imdir__ = os.path.join(im_dir)
        else:
            raise IOError("IOError! There is no folder in given path!")
        
        # reload imgs or just assign
        if ( image_obj is None ):
            im_size = (shape[1],shape[2]) # height, width
            ims = Read.readims(self.__imdir__, im_size)
            if ( scaling ):
                ims = Read.Scaling(ims)
            self.__images__ = Images( ims )
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
        return MyClassImages( self.__imdir__, new_shape, image_obj=new_image )
    
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
        
    def images(self):
        return self.__images__.__imgs__
    
    def setImages(self,np_ims_arr): # [batch,height,width,channels]
        
        self.__images__.__imgs__ = np_ims_arr
        
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
        
class Read():
    
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
    
    @staticmethod
    def Scaling(ims): # for dicom scaling
        
        # lower bound
        lower = -600
        mn = np.min(ims[:])
        if ( mn>0 ):
            lower = 0
        
        grayscale = 255
        
        ims = ims.astype('float32')
        
        mx = np.max(ims[:])
        ims[ np.where( ims < lower ) ] = np.nan
        ims = ims - lower
        ims = np.divide( ims, mx )
        ims = ims * grayscale
        ims[ np.where( np.isnan(ims) ) ] = 0

        return ims
    
    @staticmethod
    def Binaries(images): # trans all non-zero pixels to 1s and 0s otherwise.
        images[images>0] = 1
        return images
    
    
    
    


