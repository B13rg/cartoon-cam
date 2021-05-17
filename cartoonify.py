import cv2
import random as rand
from pixelsort import pixelsort
from PIL import Image
import numpy as np

from pixelsort.constants import DEFAULTS


num_down = 2 # number of downsampling steps
num_bilateral = 7 # number of bilateral filtering steps




def pixelsort_full(
    img_color, 
    upscale=False,
    mask_data=None,
    interval_image=None,
    randomness=DEFAULTS["randomness"],
    clength=DEFAULTS["clength"],
    sorting_function=DEFAULTS["sorting_function"],
    interval_function=DEFAULTS["interval_function"],
    lower_threshold=DEFAULTS["lower_threshold"],
    upper_threshold=DEFAULTS["upper_threshold"],
    angle=DEFAULTS["angle"]
):
    # Downscale first to speed things up
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # Pixelsort relies on a PIL image, to we have to convert to/from to use module.
    # Remove alpha channel of data (RGBA -> RGB)
    # Goes from (x,y,4) to (x,y,3), removing Alpha channel
    img_color = np.array(
        pixelsort(Image.fromarray(img_color, 'RGB'), 
            mask_image=mask_data, 
            interval_image=interval_image,
            randomness=randomness,
            clength=clength,
            sorting_function=sorting_function,
            interval_function=interval_function,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            angle=angle
        )
    )[:,:,:3]    
    
    if upscale:
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
    return img_color

def try_pixelsort(img_color, upscale=False):
    # Downscale first to speed things up
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # Pixelsort relies on a PIL image, to we have to convert to/from to use module.
    # Remove alpha channel of data (RGBA -> RGB)
    # Goes from (x,y,4) to (x,y,3), removing Alpha channel
    img_color = np.array(pixelsort(Image.fromarray(img_color, 'RGB')))[:,:,:3]    
    
    if upscale:
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
    return img_color

def pixelsort_Plaid(img_color, upscale=False):
    # Downscale first to speed things up
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    # Pixelsort relies on a PIL image, to we have to convert to/from to use module.
    # Remove alpha channel of data (RGBA -> RGB)
    # Goes from (x,y,4) to (x,y,3), removing Alpha channel
    img_color = np.array(pixelsort(Image.fromarray(img_color, 'RGB')))

    # Remove alpha and shift all colors, making plaid pattern
    img_color = np.resize(img_color, (img_color.shape[0], img_color.shape[1], 3))
    
    if upscale:
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
    return img_color

def colorize(img_color):
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    # upsample image to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    return img_color

def cartoonify(img_rgb):
    # downsample image using Gaussian pyramid
    img_color = try_pixelsort(img_rgb.copy(), True)

    #STEP 2 & 3
    #Use median filter to reduce noise
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    #STEP 4
    #Use adaptive thresholding to create an edge mask
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=5,
        C=2
    )

    # Step 5
    # Combine color image with edge mask & display picture
    # convert back to color, bit-AND with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)

    return img_cartoon
    # display
    #cv2.imshow("myCat_cartoon", img_cartoon)



#Select the default data source (Webcam) to acquire
cap = cv2.VideoCapture(0)

rotateInterval = 10
angle = 0
while True:
    #Get the image of each frame
    ret, frame = cap.read()
    if not ret:
        continue
    
    #Apply some effect here
    #processed = cartoonify(frame)
    #processed = try_pixelsort(frame)

    processed = pixelsort_full(frame, angle=angle)
    angle = angle + 1 % 360
    #Display image
    #I want to display to a virtual webcam
    cv2.imshow("Window Name", processed)

    #Display 30 frames and end the display when any key is pressed.
    if cv2.waitKey(30) >= 0:
        break

#End processing
cv2.destroyAllWindows()
cap.release()