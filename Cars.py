from skimage import data
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import cv2

import socket

import Utils

def main():
    #classifier_test()
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    mog2 = cv2.createBackgroundSubtractorMOG2()
    background_test(mog, 0.8)
    #background_test(mog2, 0.8)


def classifier_test():
    image_sequence = 'Data/Camera3/image_%05d.jpg'
    car_cascade = cv2.CascadeClassifier('car_classifier.xml')
    cap = cv2.VideoCapture(image_sequence)
    frame_id = 0
    while(1):
        ret, frame = cap.read()
        if ret:
            cars = car_cascade.detectMultiScale(frame)

            for (x,y,w,h) in cars:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            print 'Processing %d : cars detected : [%s]' % (frame_id, len(cars))
            cv2.imshow('frame', frame)
            cv2.waitKey(300)
            frame_id += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def background_test(bg, initial_learning):
    image_sequence = 'Data/Camera3/image_%05d.jpg'
    image_sequence = 'Images/%02d_TV388_N1PRESIDIO.jpg'
    cap = cv2.VideoCapture(image_sequence)
    back = cv2.VideoCapture(image_sequence)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    ret, frame = back.read()
    avg = np.float32(frame)

    frame_count = 0
    while(1):
        ret, frame = back.read()
        if ret:
            avg = cv2.accumulateWeighted(frame, avg, 0.05)
            frame_count = frame_count + 1
        else:
            break

    back.release()
    background = cv2.convertScaleAbs(avg)
    cv2.imshow('background',background)

    bg.apply(background, None, initial_learning)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_id = 0
    while(1):
        ret, frame = cap.read()
        if ret:
            fgmask = bg.apply(frame)



            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel, iterations=1)


            (_, contours, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #matches = []
            
            for (i, contour) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w >= 10) and (h >= 10)

                if not contour_valid:
                    continue
            
                #centroid = get_centroid(x, y, w, h)
                #cv2.rectangle(fgmask, (x,y), (x+w,y+h),(255,0,0),2)
                #cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
                #matches.append(((x, y, w, h), centroid))

            cv2.imshow('mask', fgmask)
            cv2.imshow('dilation', dilation)
            cv2.imshow('original', frame)


            orig_string = 'Output/' + str(frame_id) + '_orig.jpg'
            mask_string = 'Output/' + str(frame_id) + '_mask.jpg'
            dil_string = 'Output/' + str(frame_id) + '_dilation.jpg'
            cv2.imwrite(orig_string, frame)
            cv2.imwrite(mask_string, fgmask)
            cv2.imwrite(dil_string, dilation)

            cv2.waitKey(10)
            frame_id += 1

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def background_tests():
    image_sequence = 'Data/Camera3/image_%05d.jpg'
    cap = cv2.VideoCapture(image_sequence)
    back = cv2.VideoCapture(image_sequence)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    mog2 = cv2.createBackgroundSubtractorMOG2()

    ret, frame = back.read()
    avg = np.float32(frame)

    frame_count = 0
    while(1):
        ret, frame = back.read()
        if ret:
            avg = cv2.accumulateWeighted(frame, avg, 0.05)
            frame_count = frame_count + 1
        else:
            break
    back.release()
    background = cv2.convertScaleAbs(avg)
    cv2.imshow('background',background)

    if frame_count < 120:
        print 'Not enough frames for accurate GMG background setup'

    mog.apply(background, None, 0.8)
    mog2.apply(background, None, 0.8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_id = 0
    while(1):
        ret, frame = cap.read()
        if ret:
            fgmas_mog = mog.apply(frame)
            fgmas_mog2 = mog2.apply(frame)

            closing_mog = cv2.morphologyEx(fgmas_mog, cv2.MORPH_CLOSE, kernel)
            opening_mog = cv2.morphologyEx(closing_mog, cv2.MORPH_OPEN, kernel)
            dilation_mog = cv2.dilate(opening_mog, kernel, iterations=1)

            closing_mog2 = cv2.morphologyEx(fgmas_mog2, cv2.MORPH_CLOSE, kernel)
            opening_mog2 = cv2.morphologyEx(closing_mog2, cv2.MORPH_OPEN, kernel)
            dilation_mog2 = cv2.dilate(opening_mog2, kernel, iterations=1)

            (_, contours_mog, _) = cv2.findContours(dilation_mog, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            (_, contours_mog2, _) = cv2.findContours(dilation_mog2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for (i, contour) in enumerate(contours_mog):
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w >= 10) and (h >= 10)

                if not contour_valid:
                    continue

                centroid = get_centroid(x, y, w, h)
                cv2.rectangle(fgmas_mog, (x,y), (x+w,y+h),(255,0,0),2)

            for (i, contour) in enumerate(contours_mog2):
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w >= 10) and (h >= 10)

                if not contour_valid:
                    continue

                cv2.rectangle(fgmas_mog2, (x,y), (x+w,y+h),(255,0,0),2)

            cv2.imshow('mog', fgmas_mog)
            cv2.imshow('mog2', fgmas_mog2)

            cv2.waitKey(300)
            frame_id += 1

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

if __name__ == "__main__":
    main()