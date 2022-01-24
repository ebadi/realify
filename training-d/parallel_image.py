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



if __name__ == "__main__":
    directory = sys.argv[1]
    outdir = sys.argv[2]
    count = 0
    FILE_LIST =  directory + '/file.list'
    csvdata = {}
    platedata = {}
    with open(FILE_LIST, newline='') as csvfile:
        imgreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in imgreader:
            # print(row)

            csvdata["filename"] = row[0]
            csvdata["text"] = row[1]
            csvdata["x1"], csvdata["y1"] = row[2].split(":")
            csvdata["x2"], csvdata["y2"] = row[3].split(":")
            csvdata["x3"], csvdata["y3"] = row[4].split(":")
            csvdata["x4"], csvdata["y4"] = row[5].split(":")
            filename = csvdata["filename"]

            try:
                original_image = Image.open(os.fsdecode(directory) + '/' +filename)
            except:
                print("ERROR:: CANNOT FIND FILE:", filename)
                continue
            #### resize, keep the aspect ratio
            width = original_image.size[0]
            height = original_image.size[1]

            count = count + 1
            new_width = int(256)
            new_height = int(new_width * height / width)

            original_image_resized = original_image.resize((new_width, new_height), Image.ANTIALIAS)

            ## make it square padded
            padded = Image.new("RGB", (256, 256))
            padded.paste(original_image_resized, (0,0))


            bashCommand = "alpr " + os.fsdecode(directory) + '/' +filename + " -c eu --debug --config openalpr-plusplus.conf | grep JSON >  out/out.json"
            os.system(bashCommand)
            bashCommand = "cat out/out.json | grep DEBUG1 > out/"+ filename+ "1.json"
            # bashCommand = "cat out/out.json | grep DEBUG2 > out/"+ filename+ "2.json"

            os.system(bashCommand)
            bashCommand = "sed -e s/DEBUG1_JSON://g -i out/"+ filename+ "1.json"
            # bashCommand = "sed -e s/DEBUG2_JSON://g -i out/"+ filename+ "2.json"
            os.system(bashCommand)
            try:
                jsondata = json.load(open("out/" + filename + "1.json"))
            except:
                print("file", filename, "not selected (openALPR not detecting any license plate)")
                continue

            secondImage = Image.new("RGB", (256, 256), "WHITE")
            # secondImage.paste(original_image_resized, (0, 0))
            draw = ImageDraw.Draw(secondImage)




            x = int(csvdata["x1"])
            y = int(csvdata["y1"])
            w = int(csvdata["x3"])
            h = int(csvdata["y3"])

            drawplate(draw, x, y, w, h)

            print("File::", csvdata["filename"])
            #print("regionsOfInterest::", jsondata['regionsOfInterest'])
            #print("plateRegions::", jsondata['plateRegions'])
            #print("resultplates::")

            for plate in jsondata['resultplates']:
                #print("coords:")
                # for coord in plate['coords']:
                #    print(coord)

                #print("best:")
                for c in plate['candidates']:



                    indx = 0
                    for b in c['license']:
                        # print(b)
                        avgx1 = (b['p1x'] + b['p4x'])/2
                        avgx2 = (b['p2x'] + b['p3x'])/2

                        avgy1 = (b['p1y'] + b['p2y']) / 2
                        avgy2 = (b['p3y'] + b['p4y']) / 2

                        deltax = avgx2 - avgx1
                        deltay = avgy2 - avgy1

                        ch = b['character']

                        if indx < len(csvdata["text"]) and ch == csvdata["text"][indx] :
                            print("comparing", ch, " with ", csvdata["text"][indx])
                            writeletter(draw, avgx1, avgy1 - 2, ch)
                            csvdata["text"] = csvdata["text"][:indx] + '@' + csvdata["text"][indx + 1:]  # replace it with @ so it does not get into account next time

                        indx = indx +1

            print(csvdata["text"])
            if csvdata["text"] != "@@@@@" and csvdata["text"] != "@@@@@@" and csvdata["text"] != "@@@@@@@" and csvdata["text"] != "@@@@@@@@"  and csvdata["text"] != "@@@@@@@@@":
                continue
            double = Image.new("RGB", (512, 256))

            double.paste(padded, (0, 0))
            double.paste(secondImage, (256, 0))
            double.save(outdir + '/' + filename )

