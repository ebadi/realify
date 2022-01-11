import cv2
import pytesseract
import numpy as np
import os

directory = "."

TICKNESS = 2

def get_optimal_font_scale_height(text, height):
	for scale in reversed(range(0, 60, 1)):
		textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=TICKNESS)
		new_height = textSize[0][1]
		if (new_height <= height):
			return scale/10
	return 1

def get_optimal_font_scale_width(text, width):
	for scale in reversed(range(0, 60, 1)):
		textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=TICKNESS)
		new_width = textSize[0][0]
		if (new_width <= width):
			return scale/10
	return 1


def ocr2lic(filename):
	# read the image and get the dimensions
	img = cv2.imread(filename)
	h, w, d = img.shape # assumes color image
	blank_image = np.zeros((h,w,d), np.uint8)
	blank_image.fill(255)

	print ("h, w, d",h, w, d)
	# run tesseract, returning the bounding boxes
	boxes = pytesseract.image_to_boxes(img) # also include any config options you use
	print(boxes)
	
	x1min = 1000
	x2max = 0
	y1total = 0
	y2total = 0
	count = 0
	letter = ""
	for b in boxes.splitlines():
		b = b.split(' ')
		letter = letter + b[0]
		x1= int(b[1])
		y1= int(b[2])
		x2= int(b[3])
		y2= int(b[4])
		y1total = y1total + y1
		y2total = y2total + y2
		x1min = min(x1,x1min)
		x2max = max(x2,x2max)
		count = count +1
		img = cv2.rectangle(img, (x1, h - y1), (x2, h - y2), (0, 255, 0), 2)

	y1avg = int(y1total / count)
	y2avg = int(y2total / count)

	x1min = max( 25, x1min)
	x2max = min( w - 25, x2max)
	print("x1min, y1avg, x2max, y2avg" , x1min, y1avg, x2max, y2avg )
	img = cv2.rectangle(img, (x1min, y1avg), (x2max, y2avg), (255, 255, 0), 2)
	cv2.imshow(filename, img)

	#letter = "2AJ 4840"

	fontScalex = get_optimal_font_scale_height(letter, y2avg - y1avg)
	fontScaley = get_optimal_font_scale_width(letter, x2max - x1min)
	fontScale = min(fontScalex, fontScaley)

	font = cv2.FONT_HERSHEY_SIMPLEX
	org = (x1min, y2avg )
	color = (1, 0, 0)
	thick = 2
	blank_image = cv2.putText(blank_image, letter, org, font, fontScale, color, thick, cv2.LINE_AA)

	# show annotated image and wait for keypress
	cv2.imshow(filename, img)


	cv2.imshow("blank", blank_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


    
for filename in os.listdir("."):
	if filename.endswith(".jpg") or filename.endswith(".png"): 
		print(os.path.join(directory, filename))
		ocr2lic(os.path.join(directory, filename))
		continue
	else:
		continue




