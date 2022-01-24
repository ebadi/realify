import http.server
import socketserver
from sys import version as python_version
from cgi import parse_header, parse_multipart
# https://stackoverflow.com/questions/4233218/python-how-do-i-get-key-value-pairs-from-the-basehttprequesthandler-http-post-h/13330449
if python_version.startswith('3'):
    from urllib.parse import parse_qs
    from http.server import BaseHTTPRequestHandler
else:
    from urlparse import parse_qs
    from BaseHTTPServer import BaseHTTPRequestHandler
PORT = 8080

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                    self.rfile.read(length), 
                    keep_blank_values=1)
        else:
            postvars = {}
        return postvars

    def do_POST(self):
        postvars = self.parse_POST()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        print(postvars)
        lp = postvars[b'field1'][0].decode('UTF-8').strip().upper()
        print(lp) # execute the external program, the result will be saved in result.jpg
        import os
        os.system("cd training-d ; python3 random_image.py /home/wave/Lukas_LP/  " + lp  )
        os.system("rm /home/wave/.keras/datasets/license/test/* ; cp training-d/rnd.jpg  /home/wave/.keras/datasets/license/test/1.jpg ; python3 pix2pix.py single ~/.keras/datasets/license/; ")

        redirect = """
            <html>
              <head>
                <meta http-equiv="refresh" content="2; url='form.html'" />
              </head>
            </html>
            Loading ...
            """
        self.wfile.write(bytes(str(redirect) , encoding='utf8' ))
        return

socketserver.TCPServer.allow_reuse_address = True
httpd =  socketserver.TCPServer(("", PORT), MyHandler)
print("serving at port", PORT)
httpd.serve_forever()

