import csv
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pytesseract
IN_IMG_DIR = 'LP/'
OUT_IMG_DIR = 'LP_OUT/'
FILE_LIST= 'LP/file.list'
MARGINX = 30
MARGINY = 0
WHITE = (256, 256, 256)
BLACK = (0, 0, 0)
FONT = "din1451alt.ttf"

def parallel_image(in_dir, out_dir, img_file, text, x1, y1, x2, y2, x3, y3, x4, y4):
        print("C", img_file, text, x1, y1, x2, y2, x3, y3, x4, y4)
        minx = min(x1, x2, x3, x4)
        maxx = max(x1, x2, x3, x4)
        miny = min(y1, y2, y3, y4)
        maxy = max(y1, y2, y3, y4)
        dx = maxx - minx
        dy = maxy - miny

        original_image = Image.open(in_dir + img_file)
        width = original_image.size[0]
        height = original_image.size[1]
        new_width = int(256)
        new_height = int(new_width * height / width)
        original_resized = original_image.resize((new_width, new_height), Image.ANTIALIAS)
        #boxes = pytesseract.image_to_boxes(original_resized)
        #print("boxes",  boxes)
        original_padded = Image.new("RGB", (256, 256))
        original_padded.paste(original_resized, (0,0))

        ## New image from text
        text_image = Image.new('RGB', (256, 256), WHITE)
        draw = ImageDraw.Draw(text_image)
        fontsize = 1
        font = ImageFont.truetype(FONT, fontsize)
        while (font.getsize(text)[0] <  dx - MARGINX ) and (font.getsize(text)[1] <  dy - MARGINY )  :
            fontsize = fontsize +1
            font = ImageFont.truetype(FONT, fontsize +1 )
        print("fitting font size", fontsize)
        draw.text((minx + MARGINX, miny + MARGINY), text, BLACK, font=font)
        # text_image.save('tmp.jpg')

        double = Image.new("RGB", (512, 256))
        double.paste(text_image, (256,0))
        double.paste(original_padded, (0, 0))
        double.save(out_dir + '/' + img_file )


if __name__ == '__main__':
    with open(FILE_LIST, newline='') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in imgreader:
            #print(row)
            x1, y1 = row[2].split(":")
            x2, y2 = row[3].split(":")
            x3, y3 = row[4].split(":")
            x4, y4 = row[5].split(":")

            parallel_image( in_dir= IN_IMG_DIR, out_dir= OUT_IMG_DIR, img_file=  row[0], text= row[1],
                            x1 = int(x1),
                            y1 = int(y1),
                            x2 = int(x2),
                            y2 = int(y2),
                            x3 = int(x3),
                            y3 = int(y3),
                            x4 = int(x4),
                            y4 = int(y4),
                            )
