import cv2
import pytesseract
import numpy as np
filename = 'sample3.jpg'

def get_optimal_font_scale_height(text, height):
	for scale in reversed(range(0, 60, 1)):
		textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
		new_height = textSize[0][1]
		if (new_height <= height-10):
			return scale/10
	return 1

def get_optimal_font_scale_width(text, width):
	for scale in reversed(range(0, 60, 1)):
		textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
		new_width = textSize[0][0]
		if (new_width <= width):
			return scale/10
	return 1

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

	scale_y = get_optimal_font_scale_height(letter, y2-y1)
	scale_x = get_optimal_font_scale_width(letter, x2-x1)

	fontScale = min(scale_x, scale_y)

	# Black
	color = (0, 0, 0)
	# Line thickness of 2 px
	thickness = 2
	# Using cv2.putText() method

	blank_image = cv2.putText(blank_image, letter, org, font, fontScale, color, thickness, cv2.LINE_AA)

# show annotated image and wait for keypress
cv2.imshow(filename, img)


cv2.imshow("blank", blank_image)
cv2.waitKey(0)

