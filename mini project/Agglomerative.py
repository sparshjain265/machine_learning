import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

def saveImage (img , title, fileName):
	fig = plt.figure(figsize = (10 , 10))
	plt.imshow(img)
	plt.title (title)
	fig.savefig('Agglomerative/' + fileName)
	plt.close(fig)

images = os.listdir('image/')

count = 0

for i in images:
	count += 1
	print(count)
	#read the image
	print("Reading image: " + i)
	imgBGR = cv2.imread('image/' + i)
	imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
	# scale down for faster computation
	scale_percent = 40 # percent of original size
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

	for n in [2, 3, 4, 5]:
		print("Applying Agglomerative Clustering with " + str(n) + " clusters")
		c = cluster.AgglomerativeClustering(n_clusters=n)
		c.fit(pImage)
		labels = c.labels_
		centers = np.zeros((n, 3))
		for j in range(n):
			centers[j] = np.mean(pImage[labels == j], axis=0)
		output = centers[labels].reshape(image.shape)
		print("Saving the result")
		saveImage(output/255, "Agglomerative Clustering with " + str(n) + " clusters", i + "n=" + str(n) + ".png")