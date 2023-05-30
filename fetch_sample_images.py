import io
import os
import cv2
import tqdm
import numpy
import requests


def get_bytes(x: str):
    return numpy.frombuffer(requests.get(x).content, numpy.uint8)


def get_image(x):
    return cv2.imdecode(get_bytes(x), cv2.IMREAD_COLOR)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
# classification
# uids = """
# a784af0713a643b19ffcf65194bc0fbf
# 569a71ccf4d94c1585c9573521fb998f
# 4e6d591f6e50493aa5e31355084fc4e8
# """.split()

# caption
# uids = """
# 283c845f2c2c4567971d42dc46831372
# fc655111af5b49bf84722affc3ddba00
# fa17099f18804409bc6d9e8e397b4681
# d3c0e3495b5d40d087a7f82d1690b9cb
# 4b27adcf92f644bdabf8ecc6c5bef399
# f8c13a19e84343e7b644c19f7b9488d3
# """.split()

# sd
uids = """
b464ff8d732d44fab00b903652c8274e
efae586a477b49cea1a0777487cc2df3
f8272460c67d476a8af29e1f2e344bc0
ff2875fb1a5b4771805a5fd35c8fe7bb
b8db8dc5caad4fa5842a9ed6dbd2e9d6
tpvzmLUXAURQ7ZxccJIBZvcIDlr
""".split()


uri_fmt = 'https://objaverse-thumbnail-images.s3.us-west-2.amazonaws.com/{}.jpg'
for u in tqdm.tqdm(uids):
    img = get_image(uri_fmt.format(u))
    max_edge = max(img.shape)
    if max_edge > 512:
        s = 512 / max_edge
        img = cv2.resize(img, [0, 0], fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("samples/sd/%s.jpg" % u, img)
