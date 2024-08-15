import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
np.set_printoptions(edgeitems=2000,linewidth=2000)

#Converting Image to Numpy array
def ImageToNumpy():
    img_arr = []
    count = 0

    for file_name in os.listdir("Images/cheetah-resize-224"):

        file_path = os.path.join("Images/cheetah-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')
        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)
        count+=1

    print(len(img_arr))


    for file_name in os.listdir("Images/fox-resize-224"):

        file_path = os.path.join("Images/fox-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')
        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)

    print(len(img_arr))


    for file_name in os.listdir("Images/hyena-resize-224"):
        file_path = os.path.join("Images/hyena-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')

        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)

    print(len(img_arr))



    for file_name in os.listdir("Images/lion-resize-224"):
        file_path = os.path.join("Images/lion-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')

        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)

    print(len(img_arr))



    for file_name in os.listdir("Images/tiger-resize-224"):
        file_path = os.path.join("Images/tiger-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')

        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)

    print(len(img_arr))



    for file_name in os.listdir("Images/wolf-resize-224"):
        file_path = os.path.join("Images/wolf-resize-224", file_name)
        print(file_path)

        image = Image.open(file_path)
        image_rgb = image.convert('RGB')

        imgarr = np.array(image_rgb)

        img_arr.append(imgarr)

    print(len(img_arr))


    new_mm = np.array(img_arr)
    print(new_mm.shape)

    return new_mm



#Saving The conerted image
#np.save('img_array.npy', ImageToNumpy());