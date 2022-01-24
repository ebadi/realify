from PIL import Image, ImageFilter
from PIL import ImageFont, ImageDraw, ImageEnhance
import pymeanshift as pms
import os
import sys
import csv
import numpy

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


if __name__ == "__main__":
    directory = sys.argv[1]
    outdir = sys.argv[2]
    count = 0
    FILE_LIST =  directory + '/file.list'

    with open(FILE_LIST, newline='') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in imgreader:
            # print(row)
            x1, y1 = row[2].split(":")
            x2, y2 = row[3].split(":")
            x3, y3 = row[4].split(":")
            x4, y4 = row[5].split(":")
            filename = row[0]
            text = row[1]

            original_image = Image.open(os.fsdecode(directory) + '/' +filename)

            #### resize, keep the aspect ratio
            width = original_image.size[0]
            height = original_image.size[1]


            #draw = ImageDraw.Draw(original_image)
            #print(int(x1), int(y1), int(x4), int(y4))
            #draw.rectangle(((int(x1), int(y1)), (int(x3), int(y3))), outline=(255))
            # draw.text((20, 70), "something123", font=ImageFont.truetype("font_path123"))

            count = count + 1
            new_width = int(256)
            new_height = int(new_width * height / width)

            original_image2 = original_image.resize((new_width, new_height), Image.ANTIALIAS)

            ## make it square padded
            padded = Image.new("RGB", (256, 256))
            padded.paste(original_image2, (0,0))

            #(segmented_image, labels_image, number_regions) = pms.segment(padded, spatial_radius=6, range_radius=4.5, min_density=50)
            #pil_image=Image.fromarray(segmented_image)

            pil_imagexxxxxx = padded.convert("L")

            image = numpy.array(pil_imagexxxxxx)
            arr = binarize_array(image, 120)
            from matplotlib import cm
            pil_imagexxxxxx = Image.fromarray(cm.gist_earth(arr, bytes=True))

            pil_image = Image.new("RGB", (256, 256))
            area = (int(x1), int(y1), int(x3), int(y3))
            cropped_img = pil_imagexxxxxx.crop(area)

            #for i in range(2):
                #cropped_img = cropped_img.filter(ImageFilter.SHARPEN);
                #filter = ImageEnhance.Color(cropped_img)
                #filter.enhance(2)
            # cropped_img.show()
            pil_image.paste(cropped_img, area)
            # pil_image = pil_image.filter(ImageFilter.FIND_EDGES)

            double = Image.new("RGB", (512, 256))
            double.paste(pil_image, (256,0))
            double.paste(padded, (0, 0))
            print(count, os.fsdecode(directory), outdir + '/' + str(count) + '.jpg'  )
            double.save(outdir + '/' + str(count) + '.jpg' )
