from PIL import Image, ImageFilter
import pymeanshift as pms
import os
import sys

if __name__ == "__main__":
    directory = os.fsencode(sys.argv[1])
    outdir = sys.argv[2]
    count = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        original_image = Image.open(os.fsdecode(directory) + '/' +filename)

        #### resize, keep the aspect ratio
        width = original_image.size[0]
        height = original_image.size[1]
        if width < height :
            continue
        count = count + 1
        new_width = int(256)
        new_height = int(new_width * height / width)

        original_image2 = original_image.resize((new_width, new_height), Image.ANTIALIAS)

        ## make it square padded
        padded = Image.new("RGB", (256, 256))
        padded.paste(original_image2, (0,0))

        #(segmented_image, labels_image, number_regions) = pms.segment(padded, spatial_radius=6, range_radius=4.5, min_density=50)
        #pil_image=Image.fromarray(segmented_image)

        pil_image = padded.convert("L")
        pil_image = pil_image.filter(ImageFilter.FIND_EDGES)

        double = Image.new("RGB", (512, 256))
        double.paste(pil_image, (256,0))
        double.paste(padded, (0, 0))
        print(count, os.fsdecode(directory), outdir + '/' + str(count) + '.jpg'  )
        double.save(outdir + '/' + str(count) + '.jpg' )
