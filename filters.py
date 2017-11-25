
# coding: utf-8

# In[1]:

import numpy as np
import cv2


# In[14]:

img = cv2.imread('salt_pepper.png',1)
#img_new = corr_linear(img)
#img_new = bw(img)
#img_new = invert(img)
#img_new = box(img, 30)
#img_new = crop(img, 10, 10, 220, 120)
#img_new = corr_nonlinear(img, 1/2.2)
#img_new = wb_gw(img)
#img_new = wb_linear(img)
#img_new = gauss(img, 5)
img_new = median(img, 4)
cv2.imwrite('out.png',img_new)


# In[2]:

def corr_linear(img):
    R, G, B = cv2.split(img)
    img_max = np.max(img)
    img_min = np.min(img)
    R_new = (R - img_min)*255.0/(img_max - img_min)
    G_new = (G - img_min)*255.0/(img_max - img_min)
    B_new = (B - img_min)*255.0/(img_max - img_min)
    R_new = R_new.astype(np.uint8)
    G_new = G_new.astype(np.uint8)
    B_new = B_new.astype(np.uint8)
    img_new = cv2.merge((R_new, G_new, B_new))
    return img_new


# In[3]:

def bw(img):
    bw_matrix = np.array([[0.2125, 0.7154, 0.0721], [0.2125, 0.7154, 0.0721], [0.2125, 0.7154, 0.0721]])
    print bw_matrix
    rows = img.shape[0] 
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            img[i,j] = bw_matrix.dot(img[i,j])
    img = img.astype(np.uint8)
    return img


# In[4]:

def invert(img):
    bw_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rows = img.shape[0] 
    cols = img.shape[1]
    for i in range(rows):
        for j in range(cols):
            img[i,j] = bw_matrix.dot(img[i,j])
    img = img.astype(np.uint8)
    return img


# In[5]:

def box(img, size):
    box_matrix = np.ones((size, size))/(size**2)
    rows = img.shape[0] 
    cols = img.shape[1]
    img = cv2.copyMakeBorder(img,size/2,size/2,size/2,size/2,cv2.BORDER_REPLICATE)
    for i in range(rows):
        for j in range(cols):
            img[i,j] = np.dot(box_matrix,img[i:i+size,j]).sum(axis=0)
    img = img.astype(np.uint8)
    crop_img = img[size/2:rows, size/2:cols]
    return crop_img


# In[6]:

def crop(img, x, y, width, height):
    crop_img = img[y:y+height, x:x+width]
    return crop_img


# In[7]:

def corr_nonlinear(img, gamma):
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_new = img**float(gamma)
    return img_new


# In[8]:

def wb_gw(img):
    R, G, B = cv2.split(img)
    avg = (R.mean()+G.mean()+B.mean())/3.
    R_new = np.clip(R * avg/R.mean(), 0, 255)
    G_new = np.clip(G * avg/G.mean(), 0, 255)
    B_new = np.clip(B * avg/B.mean(), 0, 255)
    R_new = R_new.astype(np.uint8)
    G_new = G_new.astype(np.uint8)
    B_new = B_new.astype(np.uint8)
    img_new = cv2.merge((R_new, G_new, B_new))
    return img_new


# In[9]:

def wb_linear(img):
    R, G, B = cv2.split(img)
    print R
    R_new, G_new, B_new = map(lambda x: (x - np.min(x))*255.0/(np.max(x) - np.min(x)), (R, G, B))
    R_new, G_new, B_new = map(lambda x: x.astype(np.uint8), (R_new, G_new, B_new))
    print R_new
    img_new = cv2.merge((R_new, G_new, B_new))
    return img_new


# In[10]:

def gauss(img,sigma):
    kernel = gauss_function(sigma)
    size = sigma * 3
    rows = img.shape[0] 
    cols = img.shape[1]
    img = cv2.copyMakeBorder(img,size/2,size/2,size/2,size/2,cv2.BORDER_REPLICATE)
    for i in range(rows):
        for j in range(cols):
            img[i,j] = np.dot(kernel,img[i:i+size,j]).sum(axis=0)
    img = img.astype(np.uint8)
    crop_img = img[size/2:rows, size/2:cols]
    return crop_img


# In[11]:

def gauss_function(sigma):
    x = np.arange(-sigma*3 / 2 + 1., sigma*3 / 2 + 1.)
    xx, yy = np.meshgrid(x, x)
    kernel = 1./(2.*np.pi*sigma**2) * np.exp(-(xx*xx + yy*yy)/(2.*sigma**2))
    return kernel


# In[12]:

def median(img, size):
    rows = img.shape[0] 
    cols = img.shape[1]
    img = cv2.copyMakeBorder(img,size/2,size/2,size/2,size/2,cv2.BORDER_REPLICATE)
    for i in range(rows):
        for j in range(cols):
            img[i,j] = np.median(img[i:i+size, j:j+size]).sum()
    img = img.astype(np.uint8)
    crop_img = img[size/2:rows, size/2:cols]
    return crop_img


# In[ ]:



