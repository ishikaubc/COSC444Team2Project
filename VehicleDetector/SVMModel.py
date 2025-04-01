import numpy as np
import pickle
import cv2
import glob
import time

# sklearn lib
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

try:
    # sklearn > 0.17
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

from scipy.ndimage import label
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def color_convert(img_tosearch, cspace):
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img_tosearch)
    return ctrans_tosearch

def extract_features(imgs, cspace='RGB', orient=11,
                     pix_per_cell=8, cell_per_block=2,
                     hog_channel=0, size = (16, 16), hist_bins = 32, hist_range = (0, 256)):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = color_convert(image, cspace)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list

        spatial_features = bin_spatial(image, size)
        hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        features_line = np.concatenate((spatial_features, hist_features, hog_features))
        features.append(features_line)
    # Return list of feature vectors
    return features

def TrainModel():
    car_images = glob.glob('udacity_dataset/vehicles/*/*.png')
    noncar_images = glob.glob('udacity_dataset/non-vehicles/*/*.png')

    # Feature extraction parameters
    colorspace = 'YCrCb'  # 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    t = time.time()
    car_features = extract_features(car_images, cspace=colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)

    notcar_features = extract_features(noncar_images, cspace=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    # rand_state = np.random.randint(0, 100)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    # rand_state = np.random.randint(0, 100)
    rand_state = 32

    X_train_, X_test_, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    X_train = X_scaler.transform(X_train_)
    X_test = X_scaler.transform(X_test_)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))


    # Use a linear SVC
    svc = LinearSVC()


    t = time.time()
    XX = X_scaler.transform(X)
    svc.fit(XX, y)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    print('【4】Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 10))
    return svc, X_scaler

if __name__ == '__main__':
    TrainModel()