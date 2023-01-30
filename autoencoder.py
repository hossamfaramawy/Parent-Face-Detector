from __future__ import print_function, division
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.models import Model
from keras import backend as K
from data_loader import DataLoader
import datetime
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.optimizers import Adam,RMSprop
from PIL import Image as im
import matplotlib.animation as ani
#from keras.utils.vis_utils import plot_model


#interface stuff
'''
from tkinter import *
import tkinter.font as tkfont
from PIL import Image, ImageTk
import shutil
import os
from tkinter import filedialog



#     form configurations
root = Tk()
root.title("Parents Face Detector")
root.geometry("800x700")

#     fonts and styles
fontstyle_titles = tkfont.Font(size=15)
buttonfont=tkfont.Font( size=12,weight='bold')

child_uploadbutton= Button(root, text='UpLoad',font=buttonfont,padx=5, pady=5, command=pic1uploadfunction)
'''
Is_father=False

## Config 

img_h = 64
img_w = 64
channels =3 
img_shape = (img_h,img_w,channels) #64 *64 *3 width w hight w color channels RGB

def autoencoder(input_img):
    #encoder
    conv1 = Conv2D (64, (3, 3), activation='relu', padding='same')(input_img) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4) 
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) 
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5) 
    drop  = Dropout(0.5)(conv6)


    #decoder
    up1 = UpSampling2D((2,2))(drop) 
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up1) 
    up2 = UpSampling2D((2,2))(conv7) 
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(up2)  
    up3 = UpSampling2D((2,2))(conv8)  
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(up3)  
    up4 = UpSampling2D((2,2))(conv9)  
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)  
    up5 = UpSampling2D((2,2))(conv10)  

    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(up5)
    return output_img

### input Shape

input_img = Input(shape=img_shape)  #this tels the model which input or output  shape you orking with

autoencoder = Model(input_img, autoencoder(input_img))
#plot_model(autoencoder, to_file='modelploting\model1.png',show_shapes=True,show_layer_names=True)
autoencoder.summary()

#compute the frequency with which y_pred matches y_true and returns it as binary.
autoencoder.compile(optimizer = RMSprop(), loss='mean_squared_error',metrics = ['accuracy'] )

# trainer setting
epochs = 30
sample_interval = 10
batch_size=20


# Configure data loader
dataset_name = 'parent'
data_loader = DataLoader(dataset_name=dataset_name,
                                img_res=(img_h, img_w))





start_time = datetime.datetime.now()
acc=0


for epoch in range(epochs): 
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size,Is_father)): #htrg3 7agten 34ren path child , 34ren path parent w ht3mlhom enumerate bm3na hyb2a 3ndkd 34ren tuple mn (index,awlpath ,tanipath)
        #enumerate : creates a tuples in a list of the object but indexed like this [(1,'firstob'),(2,'sec')]
        # ------------------
        #  Train Generators
        # ------------------

        # Train the generators /train_on_batch takes two main param numpy array of train data and target data
        train_loss =  autoencoder.train_on_batch(imgs_A ,imgs_B)
        # returns a Scalar training loss or list of it .
        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress   -this line print the process in terminal 
        print ("[Epoch %d/%d] [Batch %d/%d] [training loss: %f, acc: %3d%%] time: %s " \
                                                                % ( epoch, epochs,
                                                                    batch_i, data_loader.n_batches,
                                                                    train_loss[0], 100*train_loss[1],
                                                                    elapsed_time))
        if 100*train_loss[1] > acc and epoch > 10 :
            autoencoder.save('./models/autoencoder_model2.h5')
            autoencoder.save_weights('./models/autoencoder_weights2.h5')
 

             
        ## Save for case of perf no inc 
        if epoch % 9==0 and batch_i == 20 :
            autoencoder.save_weights('./models/autoencoder_weights2_b_%s.h5' % (epoch))

        if batch_i % sample_interval == 0:
            # self.sample_images(epoch, batch_i)

            #os.makedirs('images/%s' % dataset_name, exist_ok=True)
            c =  3

            imgs_A = data_loader.load_data(domain="2", batch_size=1, is_testing=True) 
            imgs_B = data_loader.load_data(domain="1", batch_size=1, is_testing=True) 

            temp1=im.fromarray(imgs_A[0],'RGB')
             
            # Translate images to the other domain
            img_fake = autoencoder.predict(imgs_A)
            gen_imgs = np.concatenate([imgs_A,img_fake, imgs_B])  
            
            
            # ---my try to get the image printed or translated to pil to be used in the interface --------------------------
            finalimage=img_fake[0]
            temp=im.fromarray(finalimage,'RGB')       #temp is the image output as pil.image
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
           
            titles = ['child', 'generated parent', 'groundTruth']
            fig, axs = plt.subplots(1, c)
            cnt = 0
            for j in range(c):
                
                axs[j].imshow(gen_imgs[cnt])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
                cnt += 1
            #fig.text(0, 1,  "[Epoch %d/%d  train_loss: %3d%% time: %s ]  " % ( epoch, epochs,100*train_loss,elapsed_time), horizontalalignment='left',verticalalignment='top')
            fig.text(0, 1,  " Epoch %d/%d batch_size: %d acc: %3d%%  GL:%.4f  time: %s " % ( epoch, epochs,batch_size,100*train_loss[1],train_loss[0],elapsed_time), horizontalalignment='left',verticalalignment='top')
            
 
            plt.show()
           
          
            plt.close()
          
            #fig.savefig("C:/Users/nilso/Downloads/Compressed/Child-Face-Generator-master/Child-Face-Generator-master/output/outputimage.png")
            #fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
            
