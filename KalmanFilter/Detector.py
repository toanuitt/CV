# Import python libraries
import numpy as np
import cv2


def detect(frame, debugMode):
    # Convert frame from BGR to GRAY
    # and blur img using 7x7 gaussian filter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection using Canny function
    img_edges = cv2.Canny(blurred, 50, 190, 5)


    # Convert to black and white image
    ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 10
    max_radius_thresh= 100

    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        #Take only the valid circle(s)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))

    if (debugMode):
        cv2.imshow('gray', gray)
        cv2.imshow("img_blur", blurred)
        cv2.imshow('img_edges', img_edges)
        cv2.imshow('img_thresh', img_thresh)
        cv2.imshow('contours', img_thresh)
    return centers



