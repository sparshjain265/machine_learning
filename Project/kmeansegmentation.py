
import numpy as np
import cv2

photos = ['image2.jpg', 'image3.jpg', 'image4.jpg']
num = 2
print ("Processing image clustering......")
for each in photos:
	photo = cv2.imread(each)
	photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
	samples = photo.reshape((-1,3))
	samples = np.float32(samples)
	ct = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	#return output variables from cv2.kmeans
	# compactness, labels, centers = cv2.kmeans(samples,10,ct,10,cv2.KMEANS_RANDOM_CENTERS)
	compactness, labels, centers = cv2.kmeans(data = samples, K = 10, criteria = ct, attempts = 10,flags = cv2.KMEANS_RANDOM_CENTERS, bestLabels = None)
	centers = np.uint8(centers)
	temp = centers[labels.flatten()]
	output = temp.reshape((photo.shape))
	output = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
	output_name = 'image' + str(num) + '_output.jpg'
	#output clustered images in to separate folder
	cv2.imwrite('clusteredImages/'+output_name, output)
	num = num + 1
# cv2.destroyAllWindows()
print ("3 output images are saved in folder /clusteredImages/")
