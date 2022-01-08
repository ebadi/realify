from PIL import Image, ImageFilter
from PIL import ImageFont, ImageDraw, ImageEnhance
import pymeanshift as pms
import os
import sys
import csv
import numpy

import json
import os


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

            original_image = Image.open(os.fsdecode(directory) + '/' +filename)

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

            #print("File ", csvdata["filename"], "Text::", csvdata["text"])



            bashCommand1 = " alpr " + os.fsdecode(directory) + '/' +filename + " -c eu --debug --config openalpr-plusplus.conf | grep JSON >  out/" + filename + ".json"
            os.system(bashCommand1)

            bashCommand2 = "sed -e s/JSON://g -i out/" + filename + ".json"
            os.system(bashCommand2)
            #print(bashCommand1)
            #print(bashCommand2)

            try:
                data = json.load(open("out/" + filename + ".json"))
            except:
                print("file", filename, "not selected (openALPR not detecting any license plate)")
                pass
            platedata ={}
            for i, c in enumerate(csvdata['text']):
                platedata[i] = {}
                platedata[i]['symbol'] = csvdata['text'][i]
                platedata[i]['info'] = False

            for tid, titem  in enumerate(data['thresholds']):
                for rid, ritem in enumerate(titem['regions']):
                    for oid, oitem in enumerate(ritem['ocr_detections']):
                        for sid, sitem in enumerate(oitem['font_info']['symbols']):
                            if rid < len(platedata) and  (not platedata[rid]['info'] ) :
                                if platedata[rid]['symbol'] == sitem['symbol']:
                                    platedata[rid]["fontname"] =  oitem['font_info']["fontName"]
                                    platedata[rid]["pointsize"] = oitem['font_info']["pointsize"]
                                    platedata[rid]["x"] = ritem['x']
                                    platedata[rid]["y"] = ritem['y']
                                    platedata[rid]["height"] = ritem['height']
                                    platedata[rid]["width"] = ritem['width']
                                    platedata[rid]['info'] = True
                            if rid+1 < len(platedata) and (not platedata[rid]['info']):
                                if platedata[rid+1]['symbol'] == sitem['symbol']:
                                    platedata[rid+1]["fontname"] =  oitem['font_info']["fontName"]
                                    platedata[rid+1]["pointsize"] =  oitem['font_info']["pointsize"]
                                    platedata[rid+1]["x"] =  ritem['x']
                                    platedata[rid+1]["y"] =  ritem['y']
                                    platedata[rid+1]["height"] =  ritem['height']
                                    platedata[rid+1]["width"] =  ritem['width']
                                    platedata[rid+1]['info'] = True


                            #      The same for rid - 1 ?

            s2=""
            # print("platedata:", platedata)
            for pid in range(len(platedata)):
                s2=s2 + platedata[pid]['symbol']
            #print("symbols detected for", csvdata['text'], "other string", s2 )
            if csvdata['text'] !=  s2 :
                print("ERROR", filename)


            double = Image.new("RGB", (512, 256))
            # double.paste(pil_image, (256,0))
            double.paste(padded, (0, 0))
            #print(count, os.fsdecode(directory), outdir + '/' + filename  )
            double.save(outdir + '/' + filename )

