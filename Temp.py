import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from Apps.AppIm import MyClassImages as Cim
from Apps.AppIm import Read

shape = (3,150,150,1)
im = Cim( './imgs/posMix',shape )
print(im.shape())
im2= Cim( './imgs/posMix',shape )
print(im2.shape())
im3= im + im2
print(im3.shape())


count=0
pool = cycle(im)
pool2= cycle(im2)  

for x,y in zip(pool,pool2):
    #plt.imshow( np.squeeze(x) )
    #plt.show()
    count=count+1
    print('a'+str(x.shape))
    print('b'+str(y.shape))


print(im)
print('!')
