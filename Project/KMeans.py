import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

def saveImage (img , title, fileName):
	fig = plt.figure(figsize = (10 , 10))
	plt.imshow(img)
	plt.title (title)
	fig.savefig('KMeans/' + fileName)
	plt.close(fig)

images = os.listdir('image/')

for i in images:

	#read the image
	print("Reading image: " + i)
	imgBGR = cv2.imread('image/' + i)
	imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
	image = np.copy (imgRGB)
	saveImage(image, "original", i)

	#use median blur to remove salt noise
	image1 = cv2.medianBlur(image, 5)

	# reshape from 3D RBG to 2D
	# pImage means processed image
	pImage = image1.reshape((-1, 3))

	for n in [2, 3, 4, 5]:
		print("Applying KMeans with K = " + str(n))
		c = cluster.KMeans(n_clusters=n)
		c.fit(pImage)
		centers = c.cluster_centers_
		labels = c.labels_
		output = centers[labels].reshape(image.shape)
		print("Saving the result")
		saveImage(output/255,"K = " + str(n), i + "K=" + str(n) + ".png")