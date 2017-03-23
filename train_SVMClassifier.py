
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


#import helper_functions
#from helper_functions import extract_features

from lesson_functions import *

#Find the images files
car_dir_list = ['vehicles\GTI_Far\\', 'vehicles\GTI_Left\\', 'vehicles\GTI_Right\\', 'vehicles\GTI_MiddleClose\\', 'vehicles\KITTI_extracted\\']
noncar_dir_list = ['non-vehicles\GTI\\', 'non-vehicles\Extras\\']
car_image_list = []
noncar_image_list = []

for dir in car_dir_list:
    img_files = glob.glob(dir+'*.png')
    car_image_list.append(img_files)

for dir in noncar_dir_list:
    img_files = glob.glob(dir+'*.png')
    noncar_image_list.append(img_files)

cars = [item for sublist in car_image_list for item in sublist]
noncars = [item for sublist in noncar_image_list for item in sublist]

### DONE: Tweak these parameters and see how the results change.
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial = 16
histbin = 32
orient = 9
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = min(len(cars), len(noncars))
start_idx = 0
#print(len(cars))
#print(len(noncars))
cars = cars[start_idx:start_idx+sample_size]
notcars = noncars[start_idx:start_idx+sample_size]

#print(cars)
#print(notcars)

t=time.time()



#car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial),
#                        hist_bins=histbin, hist_range=(0, 256), orient=orient,
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                        hog_channel=hog_channel)
#notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=(spatial, spatial),
#                        hist_bins=histbin, hist_range=(0, 256), orient=orient,
#                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
#                        hog_channel=hog_channel)



car_features =  extract_features(cars, color_space=colorspace, spatial_size=(spatial,spatial),
                        hist_bins=histbin, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True)

notcar_features =  extract_features(notcars, color_space=colorspace, spatial_size=(spatial,spatial),
                        hist_bins=histbin, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
#print(X.shape)
#print(type(X))

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


# Save the SVM Classifier and the X_Scaler
import pickle

print('Writing the SVM model and Scaler model to pickle file...')
with open("model.pickle", "wb") as f:
    pickle.dump((svc, X_scaler), f)



