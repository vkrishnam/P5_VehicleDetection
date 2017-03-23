import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from scipy.ndimage.measurements import label

from math import *
import collections
from itertools import chain
from functools import reduce
from scipy.signal import find_peaks_cwt
from moviepy.editor import VideoFileClip
import sys, getopt
import os

### DONE: Tweak these parameters and see how the results change.
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial = 16
histbin = 32
orient = 9
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
y_start_stop = [350, None] # Min and max in y to search in slide_window()

import pickle
with open("model.pickle", "rb") as f:
    (svc,X_scaler) = pickle.load(f)


prev_bboxes = []

def find_vehicles_in_image(image, no_vis=True):
    global prev_bboxes
    bboxes = []
    #Medium Cars
    out_img1, bboxes = find_cars(image, 410, 530, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, (spatial, spatial),
                    histbin, bboxes=bboxes, cells_per_step = 1)
    #Small Cars
    out_img2, bboxes = find_cars(image, 390, 440, 0.25, svc, X_scaler, orient, pix_per_cell, cell_per_block, (spatial, spatial),
                    histbin, bboxes=bboxes, cells_per_step = 2)
    #out_img, bboxes = find_cars(image, 400, 500, 0.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, (spatial, spatial),
    #                histbin, bboxes=bboxes)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    #Consider the detections from the previous frame too
    #for bbox in prev_bboxes:
    #    bboxes.append(bbox)

    # Add heat to each box in box list
    heat_img = add_heat(heat,bboxes)
    heat_img = add_heat(heat_img,prev_bboxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat_img,2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, bboxes = draw_labeled_bboxes(np.copy(image), labels)
    prev_bboxes = bboxes

    if no_vis is False:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        #fig.tight_layout()

    #return draw_boxes(image, bboxes)
    #return out_img
    #return heat_img
    return draw_img

#process the video file
def process_image(image):
    return find_vehicles_in_image(image, no_vis=True)


def main(argv):
    inputVideoFile = None
    outputVideoFile = None
    global cam_dist, cam_mtx
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('vehicle_detection_pipeline.py -i <inputVideoFile> -o <outputVideoFile> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('vehicle_detection_pipeline.py -i <inputVideoFile> -o <outputVideoFile>  ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputVideoFile = arg
        elif opt in ("-o", "--ofile"):
            outputVideoFile = arg
    print( 'Input Video file is "', inputVideoFile)
    print( 'Output Video file is "', outputVideoFile)

    if (inputVideoFile is None) | (outputVideoFile is None):
        #findLanesInTestImages()
        sys.exit(0)

    #Do the video processing
    outfile = outputVideoFile
    clip1 = VideoFileClip(inputVideoFile)
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(outfile, audio=False)
    sys.exit(0)


if __name__ == "__main__":
   main(sys.argv[1:])


