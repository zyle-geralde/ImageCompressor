import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.cluster import KMeans

#image sample data
image = Image.open("Images/cheetah-resize-224/00000000_224resized.png")

#convert to RGB(3 dimensions)
image_rgb = image.convert('RGB')


#convert to numpy array
imgarr = np.array(image_rgb)
print(imgarr)

#normalize pixel value range(0-1)
imgarr = imgarr/255
print(imgarr)

#reshape to 2 dimensions
reshaped_image = np.reshape(imgarr,(imgarr.shape[0] * imgarr.shape[1],imgarr.shape[2]))
print(reshaped_image.shape)


#Kmeans algorithm






'''reshaped_data = np.reshape(alldata,(alldata.shape[0],alldata.shape[1] * alldata.shape[2],alldata.shape[3]))
print(reshaped_data.shape)'''







