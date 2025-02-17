import cv2
import numpy as np
from google.colab import files
from IPython.display import display, Image

# Upload the image
uploaded = files.upload()

# Process the uploaded image
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

  # Read the image using OpenCV
  img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_UNCHANGED)

  # Convert the image to grayscale
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Convert to black and white using thresholding
  _, bw_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

  # Detect edges from the black and white image
  edges = cv2.Canny(bw_img, 100, 200)  # Adjust thresholds as needed

  # Display the results
  print("Original Image:")
  display(Image(data=uploaded[fn]))  # Display original image
  print("\nBlack and White Image:")
  display(Image(data=cv2.imencode('.png', bw_img)[1].tobytes()))  # Display B&W image
  print("\nEdge Detected Image:")
  display(Image(data=cv2.imencode('.png', edges)[1].tobytes()))  # Display edges