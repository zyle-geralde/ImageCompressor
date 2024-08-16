import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from PIL import Image
import os
from sklearn.cluster import KMeans

#image sample data
image = Image.open("Images/cheetah-resize-224/00000002_224resized.png")

#convert to RGB(3 dimensions)
image_rgb = image.convert('RGB')


#convert to numpy array
imgarr = np.array(image_rgb)

#normalize pixel value range(0-1)
imgarr = imgarr/255

#reshape to 2 dimensions
reshaped_image = np.reshape(imgarr,(imgarr.shape[0] * imgarr.shape[1],imgarr.shape[2]))


#Kmeans algorithm

km = KMeans(n_clusters=16,init='k-means++', n_init=50, max_iter=500, random_state=42)

y_predicted = km.fit_predict(reshaped_image)

#clusters (16 clusters)
colors = km.cluster_centers_


#showing the 16 colors(centroids) generated
fi,ax = plt.subplots(1,1,figsize = (colors.shape[0],1))
for i, color in enumerate(colors):
    rect = plt.Rectangle((i, 0), 1, 1, color=color)
    ax.add_patch(rect)

# Adjust the plot limits and remove axes
#x should range from 0 to the number of colors
#this will have 16 columns since number of colors = 16
ax.set_xlim(0, colors.shape[0])
#this will have 1 row
#y should range from 0 and 1
ax.set_ylim(0, 1)

ax.axis('off')

# Display the plot
#plt.show()

#list of the centroids to its associated x inputs
centroid_labels = km.labels_
# Assign each pixel to the closest centroid
compressed_imgarr = colors[centroid_labels]

#turn the image back to its original shape
compressed_imgarr = compressed_imgarr.reshape(imgarr.shape)

#convert pixel values back to range(0,255)
compressed_imgarr = (compressed_imgarr*255).astype(np.uint8)

#show compressed image
compressed_imgarr = Image.fromarray(compressed_imgarr)
compressed_imgarr.show()
#show original image
orgimage = (imgarr*255).astype(np.uint8)
orgimage = Image.fromarray(orgimage)
orgimage.show()




'''reshaped_data = np.reshape(alldata,(alldata.shape[0],alldata.shape[1] * alldata.shape[2],alldata.shape[3]))
print(reshaped_data.shape)'''







