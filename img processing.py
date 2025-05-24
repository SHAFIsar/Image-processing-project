import cv2
import numpy as np
from google.colab import files
from IPython.display import display, Image

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

  img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_UNCHANGED)

  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  _, bw_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

  edges = cv2.Canny(bw_img, 100, 200)

  print("Original Image:")
  display(Image(data=uploaded[fn]))
  print("\nBlack and White Image:")
  display(Image(data=cv2.imencode('.png', bw_img)[1].tobytes()))
  print("\nEdge Detected Image:")
  display(Image(data=cv2.imencode('.png', edges)[1].tobytes()))