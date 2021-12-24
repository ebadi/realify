import cv2
import pytesseract
import numpy as np
filename = 'sample3.jpg'

# read the image and get the dimensions
img = cv2.imread(filename)
h, w, d = img.shape # assumes color image
blank_image = np.zeros((h,w,d), np.uint8)
blank_image.fill(255)


# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use
print(boxes)

# draw the bounding boxes on the image
for b in boxes.splitlines():
	b = b.split(' ')
	letter = b[0]
	x1= int(b[1])
	y1= int(b[2])
	x2= int(b[3])
	y2= int(b[4])


	img = cv2.rectangle(img, (x1, h - y1), (x2, h - y2), (0, 255, 0), 2)

	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
	# org
	org = (x1, y1)
	# fontScale
	fontScale = 0.5
	# Black
	color = (0, 0, 0)
	# Line thickness of 2 px
	thickness = 1
	# Using cv2.putText() method
	blank_image = cv2.putText(blank_image, letter, org, font, fontScale, color, thickness, cv2.LINE_AA)

# show annotated image and wait for keypress
cv2.imshow(filename, img)


cv2.imshow("blank", blank_image)
cv2.waitKey(0)

