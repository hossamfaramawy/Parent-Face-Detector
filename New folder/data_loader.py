import scipy.misc
from glob import glob
import numpy as np
import scipy.misc.pilutil as sc

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        if(is_testing==False):
            data_type = "train"
        else: 
            data_type = "test"
        path = glob('./datasets/%s/%s/*%s.jpg' % (self.dataset_name, data_type,domain)) #returns list of path names that matches 
        
        batch_images = np.random.choice(path, size=batch_size)
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            '''
            if  is_testing==False:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                '''
            img = sc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False , is_father=True):
        if(is_testing==False):
            data_type = "train"
        else: data_type = "test"  
        
        if(is_father==True):
            predicted_parent="father-child"
        else: 
            predicted_parent="mother_child"
        
        
        path_A = glob('./datasets/%s/%s/%s/*2.jpg' % (self.dataset_name, data_type,predicted_parent)) # hyload kol soar al childs
        path_B = glob('./datasets/%s/%s/%s/*1.jpg' % (self.dataset_name, data_type,predicted_parent)) # hyload kol soar al parents
        #34n na5od al 3dd zogy dyman
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size) #3dd al 22l l2nna bn4t8l p w c fa  hna5od al 3dd al 22l w n2smo 3la al size al batch al wa7d hyt7dd 3dd al batchat bs kda
        #total_samples = self.n_batches * batch_size # 34an lo fe batch na2s mn5do4 bl samples w n7sb al samples al kamla bs blnsba ll batches al kamla

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        '''
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        '''
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size] #ym4y 20 20 kol mara 20 godad l8ayt ma y5lshom
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B): #zip:return a iterable of tuples (i of list1,i oflist2) i=0 to listsize, hna imagea w b htb2a feha awl sorten mn al tulple al ola w hakza k2nk bt2olo i,b mslan from awl path (path1,path2)
                img_A = self.imread(img_A) #hy3ml read l2ol path f tuple aly wa2f 3ndha
                img_B = self.imread(img_B) #hy3ml read ltany path f tuple

                img_A = sc.imresize(img_A, self.img_res)
                img_B = sc.imresize(img_B, self.img_res)
                # makes sure that the batch had random pics in it
                '''
                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                '''
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.  # np array is turning the image to 3d array, w da m3nah hyb2a feh value lkol pixel fl r wl b wl g  fa hn2sm al value de w ntr7 mnha 1
            imgs_B = np.array(imgs_B)/127.5 - 1.  # awl 34reen image hyt3mlhom normalising to range [-1 to 1 ]

            yield imgs_A, imgs_B #yield: runs from the last state from where the function is paused , can run multiple times and , hna hyrun awl for w y3ml pause w yrg3 al result w ysave al state w b3dyn ,lma yt3ml call ykml tany for m4 mn al awl bybd2 mn a5er state w btb2a 7afza al local variables wkol da 7lo ll memory kman. 
            

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return sc.imread(path, mode='RGB').astype(np.float)