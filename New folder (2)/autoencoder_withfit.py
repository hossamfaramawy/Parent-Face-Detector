# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:59:35 2021

@author: nilso
train with fiting 

"""  
   
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
from tensorflow import keras
import scipy.misc
from glob import glob
import numpy as np
import scipy.misc.pilutil as sc
#from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import load_model


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
#constructor to configure the data 

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

#functions of the model here 



# trainer setting
epochs = 1000
sample_interval = 10
batch_size=20

#configure data loader 
dataset_name = 'parent'
data_loader = DataLoader(dataset_name=dataset_name,
                                img_res=(img_h, img_w))

"""

#-------------------------train type------------------------
train_type="father-child"
#train_type="mother-child"

#load the data before using it in a list of all pathes child and parent
pathchild = glob('./datasets/parent/train/%s/*2.jpg'%(train_type))
pathparent = glob('./datasets/parent/train/%s/*1.jpg'%(train_type))

#x_train:childs of train,y_train: parent of train pathes,and the same for test
x_train, x_test, y_train, y_test = train_test_split(pathchild, pathparent)


#get images ready in list.
x_trainlist, x_testlist, y_trainlist, y_testlist =[],[],[],[]


for img_child, img_parent in zip(x_train , y_train):
    img_child_temp = data_loader.imread(img_child) 
    img_parent_temp = data_loader.imread(img_parent) 
    img_A = sc.imresize(img_child_temp, data_loader.img_res)
    img_B = sc.imresize(img_parent_temp, data_loader.img_res)
    #each image is appended to the list as an image of [3d list] like that [[3d list of image 1],[3d list of image 2], ....]
    x_trainlist.append(img_A)
    y_trainlist.append(img_B)
#scaling the images to [-1,1] after appending them and putting them as 1 array   
x_train = np.array(x_trainlist)/127.5 - 1.
y_train = np.array(y_trainlist)/127.5 - 1.


for img_child, img_parent in zip(x_test , y_test):
    img_child_temp = data_loader.imread(img_child) 
    img_parent_temp = data_loader.imread(img_parent) 
    img_A = sc.imresize(img_child_temp, data_loader.img_res)
    img_B = sc.imresize(img_parent_temp, data_loader.img_res)
    #each image is appended to the list as an image of [3d list] like that [[3d list of image 1],[3d list of image 2], ....]
    x_testlist.append(img_A)
    y_testlist.append(img_B)
#scaling the images to [-1,1] after appending them and putting them as 1 array   
x_test = np.array(x_trainlist)/127.5 - 1.
y_test = np.array(y_trainlist)/127.5 - 1.

#train the model 
history=autoencoder.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


#--------------------------save the model and weights--------------------------
autoencoder.save('./models/autoencoder_model200.h5')
autoencoder.save_weights('./models/autoencoder_weights200.h5')
 """
#--------------------------load the modela and weights-------------------------
# reasoning is that h5py.File(path) doesn't handle the standard subfolder path so the model must be loaded from the main directory of the project or it will not be loaded 
autoencoder_loaded = load_model('./models/autosavedmodel1000.h5')
#autoencoder_loaded.close()

#-------------------preparation for prediction  ----------------
imgs=[]
input_img_predict = data_loader.imread('./datasets/parent/test/ss.jpg')
input_img_resized = sc.imresize(input_img_predict, data_loader.img_res)
imgs.append(input_img_resized)
input_img_normalized = np.array(imgs)/127.5 - 1.


#prediction and unscaling the image 
imgfake = autoencoder_loaded.predict(input_img_normalized)
#imgfake= autoencoder.predict(input_img_normalized)
gen_imgs = 0.5 * imgfake + 0.5


#-------------converting the image from numpy array to image----------------
final_image=gen_imgs[0] # rgb (64,64,3)
#final_image.show()


# the solution because after prediction the shape was (1,64,64,3) then i took the (64,64,3) by the gen[0]
#and its now a numpy array of this (64,64,3) ,and there is a value problem when i try to convert it to
#image because its value is float 0.nine num or something,so by multipling it become normal num between 255 and 0 but float
#as np.unit8 translate them to whats equal to their values but int and then we can use fromarray to translate it to pil image.


#converting image to pil image and unscaling it to 255 rgb intger
final_image=final_image*255.0
final_image=final_image.astype(np.uint8)
f=im.fromarray(final_image)
f.save("test_new"+str(epochs)+".png")
#f.show()
#-----------------working plot just for testing -----------------

c=1
titles = ['generated parent']
fig, axs = plt.subplots(1, 2)
cnt = 0
for j in range(c):
    
    axs[j].imshow(gen_imgs[cnt])
    axs[j].set_title(titles[j])
    axs[j].axis('off')
    cnt += 1
plt.show()          
plt.close()
