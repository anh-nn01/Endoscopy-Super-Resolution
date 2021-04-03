"""
Created on Wed Mar, 3, 2021
@author: Nhu Nhat Anh

Deploy Demo
"""
import cv2
import os

from File_Browser import file_browser


def click_and_crop(event, x, y, flags, param):
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    
    # image = clone.copy()
    if (event == cv2.EVENT_LBUTTONDOWN):
        """
        Patch retrieval
        """
        patch = image[y-127:y+128, x-127:x+128]
        cv2.imshow("Patch", patch)
        patch_PIL = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) 
        
        cv2.rectangle(image, (x-127, y-127), (x+128, y+128), color = (0,255,0), thickness = 2)       
        cv2.imshow("Image", image)
#####################################################################################################################################################

# Load the image, clone it, and setup the mouse callback function
# img_dir = "/home/computer1/Downloads/Med/Sélection d_images D.Lamarque Partie I/Gastrite atrophique/Image_283416.png"
img_dir = file_browser()
image = cv2.imread(img_dir)
clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_crop)

loop = True
# keep looping until the Esc key is pressed
while loop:
    # display the image and wait for a keypress
    cv2.imshow("Image", image)      
    key = cv2.waitKey(0) & 0xFF
    image = clone.copy()
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r") or key == 8:
        image = clone.copy()
    # if the Esc key is pressed, break from the loop
    if key == 27:
        loop = False
        cv2.destroyAllWindows()
