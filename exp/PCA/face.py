from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

faceImage=Image.open(r"exp\PCA\1.BMP")

faceMat=np.asarray(faceImage)

print(faceMat.shape)

image=Image.fromarray(faceMat)
image.save("res.BMP")
plt.figure()
plt.imshow(image)
plt.show()