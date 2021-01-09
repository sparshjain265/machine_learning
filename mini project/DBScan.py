import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

def saveImage (img , title, fileName):
	fig = plt.figure(figsize = (10 , 10))
	plt.imshow(img)
	plt.title (title)
	fig.savefig('DBScan/' + fileName)
	plt.close(fig)

images = os.listdir('image/')

for i in images:

	#read the image
	print("Reading image: " + i)
	imgBGR = cv2.imread('image/' + i)
	imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
	# scale down for faster computation
	scale_percent = 50 # percent of original size
	width = int(imgRGB.shape[1] * scale_percent / 100)
	height = int(imgRGB.shape[0] * scale_percent / 100)
	dim = (width, height)
	img = cv2.resize(imgRGB, dim, interpolation = cv2.INTER_AREA)
	image = np.copy (img)
	saveImage(image, "resized", i)

	#use median blur to remove salt noise
	image1 = cv2.medianBlur(image, 5)

	# reshape from 3D RBG to 2D
	# pImage means processed image
	pImage = image1.reshape((-1, 3))

	for e in [1.0, 1.5, 2.0, 2.5]:
		print("Applying DBScan with e = " + str(e))
		c = cluster.DBSCAN(eps = e)
		c.fit(pImage)
		labels = c.labels_
		# print(np.max(labels))
		# print(np.min(labels))
		n = np.max(labels) + 2
		centers = np.zeros((n, 3))
		for j in range(n):
			centers[j] = np.mean(pImage[labels == j-1], axis=0)
		output = centers[labels].reshape(image.shape)
		print("Saving the result")
		saveImage(output/255, "DBScan with epsilon: " + str(e) + " gives " + str(n) + " clusters", i + "e=" + str(e) + ".png")