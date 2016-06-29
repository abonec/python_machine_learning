import skimage
from sklearn.cluster import KMeans
from skimage.io import imread
import pylab as plt


image = imread('parrots.jpg')

float_image = skimage.img_as_float(image)

X = float_image.reshape((float_image.shape[0] * float_image.shape[1], float_image.shape[2]))


model = KMeans(init='k-means++', random_state=241)

model.fit(X)


