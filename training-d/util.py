from PIL import ImageFont, ImageDraw, ImageEnhance
FONT = 'din1451alt.ttf'

def find_font_size(ch, deltax, deltay):
    fontsize = 31
    font = ImageFont.truetype(FONT, fontsize)
    while font.getsize(ch)[0] < deltax and font.getsize(ch)[1] < deltay:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(FONT, fontsize + 1)
    #print("font::::", fontsize)
    return fontsize

def drawplate(draw, x, y, w, h):
    bar = 7
    draw.line((x + bar, y, x + bar, h), fill=(0, 255, 0), width=bar * 2)
    draw.line((w, y, w, h), fill=(0, 0, 255), width=2)

    draw.line((x, y, w, y), fill=(255, 0, 0), width=2)
    draw.line((x, h, w, h), fill=(0, 255, 255), width=2)

def writeletter(draw, x,y,ch):
    fontsize = find_font_size(ch, x, y)
    font = ImageFont.truetype(FONT, fontsize)
    # optionally de-increment to be sure it is less than criteria
    draw.text((x, y - 2), ch, font=font, fill="BLACK")
