import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
image = Image.open("Images/cheetah-resize-224/00000000_224resized.png")

npimg = np.array(image)
print(npimg.shape)