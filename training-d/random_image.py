from PIL import Image, ImageFilter
from PIL import ImageFont, ImageDraw, ImageEnhance
import pymeanshift as pms
import os
import sys
import csv
import numpy

import json
import os
FONT = 'din1451alt.ttf'
import util
from util import *
import glob
import random
NOISE = 1
random.seed()
if __name__ == "__main__":
    directory = sys.argv[1]
    licenseplate = sys.argv[2]

    rndid = random.randrange(15000)
    FILE_LIST =  directory + '/file.list'
    csvdata = {}
    with open(FILE_LIST, newline='') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        c = 0
        for row in imgreader:
            # print(row)
            c = c +1
            csvdata["filename"] = row[0]
            csvdata["text"] = row[1]
            csvdata["x1"], csvdata["y1"] = row[2].split(":")
            csvdata["x2"], csvdata["y2"] = row[3].split(":")
            csvdata["x3"], csvdata["y3"] = row[4].split(":")
            csvdata["x4"], csvdata["y4"] = row[5].split(":")
            rndfile = directory + "/" +  csvdata["filename"]
            if c > rndid and len(csvdata["text"]) == len(licenseplate) :
                break


    original_image = Image.open(rndfile)
    #### resize, keep the aspect ratio
    width = original_image.size[0]
    height = original_image.size[1]
    new_width = int(256)
    new_height = int(new_width * height / width)
    original_image_resized = original_image.resize((new_width, new_height), Image.ANTIALIAS)
    padded = Image.new("RGB", (256, 256))
    padded.paste(original_image_resized, (0, 0))
    bashCommand = "alpr " + rndfile + " -c eu --debug --config openalpr-plusplus.conf | grep JSON >  rnd.json"
    os.system(bashCommand)
    bashCommand = "cat rnd.json | grep DEBUG1 > rndfiltered.json"
    os.system(bashCommand)
    bashCommand = "sed -e s/DEBUG1_JSON://g -i rndfiltered.json"
    os.system(bashCommand)

    jsondata = {}
    try:
        jsondata = json.load(open("rndfiltered.json"))
    except:
        print("file", rndfile, "not selected (openALPR not detecting any license plate)")

    secondImage = Image.new("RGB", (256, 256), "WHITE")
    draw = ImageDraw.Draw(secondImage)
    x = int(csvdata["x1"])
    y = int(csvdata["y1"])
    w = int(csvdata["x3"])
    h = int(csvdata["y3"])

    print(x,y,w,h)
    drawplate(draw, x + random.randint(-1 * NOISE, NOISE) , y + random.randint(-1 * NOISE, NOISE) , w + random.randint(-1 * NOISE, NOISE), h + random.randint(-1 * NOISE, NOISE))

    for plate in jsondata['resultplates']:
            indx = 0
            for b in plate['best']:

                # print(b)
                avgx1 = (b['p1x'] + b['p4x']) / 2
                avgx2 = (b['p2x'] + b['p3x']) / 2

                avgy1 = (b['p1y'] + b['p2y']) / 2
                avgy2 = (b['p3y'] + b['p4y']) / 2

                deltax = avgx2 - avgx1
                deltay = avgy2 - avgy1

                ch = b['character']
                ch = licenseplate[indx]

                # optionally de-increment to be sure it is less than criteria
                writeletter(draw, avgx1 + random.randint(-1 * NOISE, NOISE) , avgy1 + random.randint(-1 * NOISE, NOISE), ch)
                indx = indx + 1


    double = Image.new("RGB", (512, 256))

    double.paste(padded, (0, 0))
    double.paste(secondImage, (256, 0))
    double.save("rnd.jpg")

    exit()
