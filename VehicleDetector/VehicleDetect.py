import numpy as np
import pickle
import cv2
import glob
import time

# sklearn lib
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib  # 新版
import pickle
import SVMModel

import SVMModel

try:
    # sklearn > 0.17
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

from scipy.ndimage import label
from skimage.feature import hog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip #pip install moviepy==1.0.3
from IPython.display import HTML
import SVMModel

print(f'libraries import successful!')

G_Realtime_training = False
# G_SVM = loaded_clf = joblib.load("models/SVM_VehicleDetection.pkl")
# G_X_Scaler = joblib.load("models/X_Scaler_VehicleDetection.pkl")
G_SVM = None
G_X_Scaler = None
G_MSVM = None
G_MX_Scaler = None
if G_Realtime_training:
    G_SVM, G_X_scaler = SVMModel.TrainModel()
else:
    with open("models/SVM_VehicleDetection.pkl", "rb") as f:
        G_SVM = pickle.load(f)
    with open("models/X_Scaler_VehicleDetection.pkl", "rb") as f:
        G_X_Scaler = pickle.load(f)
    G_MSVM = joblib.load("models/MSVM_VehicleDetect.pkl")
    print(f'G_MSVM = {G_MSVM}')
    G_MX_Scaler = joblib.load("models/MX_Scaler_VehicleDetect.pkl")
    print(f'G_MX_Scaler = {G_MX_Scaler}')
print(f'G SVM: {G_SVM}')
print(f'X Scaler: {G_X_Scaler}')


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = int(xy_window[1] * (xy_overlap[1]))
    nx_windows = int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

G_Predict_Stat = {"totalTime":0,"totalCount":0.0}
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins,
              show_all=False, size = (16, 16), hist_range = (0, 256)):
    print(f'Finding cars with paras: ystart={ystart}, ystop={ystop}, scale={scale}, cspace={cspace}, hog_channel={hog_channel}, orient={orient}, pix_per_cell={pix_per_cell}, show_all={show_all}')
    # array of rectangles where cars were detected
    rectangles = []
    scores = []
    hist_bins = 32

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion if other than 'RGB'
    ctrans_tosearch = SVMModel.color_convert(img_tosearch, cspace)

    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1] / scale), int(imshape[0] / scale)))

    # select colorspace channel for HOG
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
        if ch1.shape[0] < 32 or ch1.shape[1] < 32:
            return [], []
        if ch2.shape[0] < 32 or ch2.shape[1] < 32:
            return [], []
        if ch3.shape[0] < 32 or ch3.shape[1] < 32:
            return [], []
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]
        if ch1.shape[0] < 32 or ch1.shape[1] < 32:
            return [], []



    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = SVMModel.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = SVMModel.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = SVMModel.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            start_time = time.time()
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            spatial_features = SVMModel.bin_spatial(subimg, size)
            hist_features = SVMModel.color_hist(subimg, nbins=hist_bins, bins_range=hist_range)
            features_line = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            features = X_scaler.transform(features_line)
            # print(f'feature size: {features.shape}')
            test_prediction = svc.predict(features)
            score = svc.decision_function(features)

            end_time = time.time()
            seconds = end_time - start_time
            G_Predict_Stat['totalTime'] += seconds
            G_Predict_Stat['totalCount'] += 1
            print(f'[SVM]Elapsed time: {seconds:.2f} seconds, average time {G_Predict_Stat["totalTime"] / G_Predict_Stat["totalCount"]}, total inference count: {G_Predict_Stat["totalCount"]}')

            if test_prediction == 1 or show_all:
                xbox_left = int(xleft * scale)
                ytop_draw = int(ytop * scale)
                win_draw = int(window * scale)
                rectangles.append(((xbox_left, ytop_draw + ystart),
                                   (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                scores.append(score)

    return rectangles, scores

def find_cars_M(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins,
              show_all=False, categories = ["Non-Vehicles", "Sedan", "Semi", "SUV", "Truck"], size = (16, 16), hist_range = (0, 256)):
    print(f'[M]Finding cars with paras: ystart={ystart}, ystop={ystop}, scale={scale}, cspace={cspace}, hog_channel={hog_channel}, orient={orient}, pix_per_cell={pix_per_cell}, show_all={show_all}, categories={categories}')
    # return values
    rectangles = []
    scores = []
    tags = []
    hist_bins = 32

    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion if other than 'RGB'
    ctrans_tosearch = SVMModel.color_convert(img_tosearch, cspace)

    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (int(imshape[1] / scale), int(imshape[0] / scale)))

    # select colorspace channel for HOG
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
        if ch1.shape[0] < 32 or ch1.shape[1] < 32:
            return [], [], []
        if ch2.shape[0] < 32 or ch2.shape[1] < 32:
            return [], [], []
        if ch3.shape[0] < 32 or ch3.shape[1] < 32:
            return [], [], []
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]
        if ch1.shape[0] < 32 or ch1.shape[1] < 32:
            return [], [], []



    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = SVMModel.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = SVMModel.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = SVMModel.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            start_time = time.time()
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            spatial_features = SVMModel.bin_spatial(subimg, size)
            hist_features = SVMModel.color_hist(subimg, nbins=hist_bins, bins_range=hist_range)
            features_line = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)

            features = X_scaler.transform(features_line)

            ret = svc.predict(features)
            prediction = int(ret[0])
            all_prob = svc.predict_proba(features)
            probabilities = all_prob[0]

            end_time = time.time()
            seconds = end_time - start_time
            G_Predict_Stat['totalTime'] += seconds
            G_Predict_Stat['totalCount'] += 1
            print(f'[SVM]Elapsed time: {seconds:.2f} seconds, average time {G_Predict_Stat["totalTime"] / G_Predict_Stat["totalCount"]}, total inference count: {G_Predict_Stat["totalCount"]}')

            # print(f"预测类别: {categories[prediction]}")
            # print("类别概率:")
            max_prob = 0
            max_prob_idx = 0
            max_tag = ''
            for i, category in enumerate(categories):
                if probabilities[i] > max_prob:
                    max_prob = probabilities[i]
                    max_prob_idx = i
                    max_tag = category
                # print(f"{category}: {probabilities[i]:.4f}")

            if max_tag != 'Non-Vehicles' or show_all:
                xbox_left = int(xleft * scale)
                ytop_draw = int(ytop * scale)
                win_draw = int(window * scale)
                rectangles.append(((xbox_left, ytop_draw + ystart),
                                   (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                scores.append(max_prob)
                tags.append(max_tag)

    return rectangles, scores, tags

G_Process_Stat = {"totalTime":0,"totalCount":0.0}
def process_frame(img, confidence_threshold=0.8):
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    # ystarts = [400, 500]
    ystarts = [int(img.shape[0]*0.2), int(img.shape[0]*0.5)]
    # ystarts = [i for i in range(300, 700, 200)]
    # ystarts = [i for i in range(int(img.shape[0]*0.3), int(img.shape[0]*0.6), 100)]
    y_scale = [1, 1.5, 2, 2.5, 3]
    #y_step = [i * 16 for i in y_scale]
    #print(y_step)
    y_stop = img.shape[0]# 800
    start_time = time.time()
    print(f'process_frame(): img shape: {img.shape}, ystarts: {ystarts}, y_scale: {y_scale}, y_stop: {y_stop}')

    #ystop_arr = [i +  for i in ystart_arr]
    #show_all_rectangles=True
    show_all_rectangles=False
    rectangles = []
    scores = []
    for i in range(len(y_scale)):
        for j in range(len(ystarts)):
            scale = y_scale[i]
            ystart = ystarts[j]
            step = int((y_stop - ystart)//(scale * pix_per_cell) )
            ystop = int(ystart + step * scale * pix_per_cell)
            #print(scale, step, ystart, ystop)
            rects, ss = find_cars(img, ystart, ystop, scale, colorspace, hog_channel, G_SVM, G_X_Scaler,
                               orient, pix_per_cell, cell_per_block, None, None, show_all_rectangles)
            rectangles.append(rects)
            scores.append(ss)

    rectangles = [item for sublist in rectangles for item in sublist]
    scores = [item for sublist in scores for item in sublist]
    final_rectangles = []
    final_scores = []
    for i in range(0, len(rectangles)):
        if scores[i] >= confidence_threshold:
            final_rectangles.append(rectangles[i])
            final_scores.append(scores[i])

    end_time = time.time()
    seconds = end_time - start_time
    G_Process_Stat['totalTime'] += seconds
    G_Process_Stat['totalCount'] += 1
    print(f'[process_frame]Elapsed time: {seconds:.2f} seconds, average time {G_Process_Stat["totalTime"] / G_Process_Stat["totalCount"]}, total inference count: {G_Process_Stat["totalCount"]}')

    return final_rectangles, final_scores


def process_frame_M(img, confidence_threshold=0.8):
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    # ystarts = [400, 500]
    ystarts = [int(img.shape[0]*0.4), int(img.shape[0]*0.5)]
    # ystarts = [i for i in range(300, 700, 200)]
    # ystarts = [i for i in range(int(img.shape[0]*0.3), int(img.shape[0]*0.6), 100)]
    y_scale = [1, 1.5, 2, 2.5, 3]
    #y_step = [i * 16 for i in y_scale]
    #print(y_step)
    y_stop = img.shape[0]#800
    start_time = time.time()
    print(f'process_frame(): img shape: {img.shape}, ystarts: {ystarts}, y_scale: {y_scale}, y_stop: {y_stop}')

    #ystop_arr = [i +  for i in ystart_arr]
    #show_all_rectangles=True
    show_all_rectangles=False
    rectangles = []
    scores = []
    tags = []
    for i in range(len(y_scale)):
        for j in range(len(ystarts)):
            scale = y_scale[i]
            ystart = ystarts[j]
            step = int((y_stop - ystart)//(scale * pix_per_cell) )
            ystop = int(ystart + step * scale * pix_per_cell)
            #print(scale, step, ystart, ystop)
            rects, ss, tag = find_cars_M(img, ystart, ystop, scale, colorspace, hog_channel, G_MSVM, G_MX_Scaler,
                               orient, pix_per_cell, cell_per_block, None, None, show_all_rectangles)
            rectangles.append(rects)
            scores.append(ss)
            tags.append(tag)

    rectangles = [item for sublist in rectangles for item in sublist]
    scores = [item for sublist in scores for item in sublist]
    tags = [item for sublist in tags for item in sublist]
    final_rectangles = []
    final_scores = []
    final_tags = []
    for i in range(0, len(rectangles)):
        if scores[i] >= confidence_threshold:
            final_rectangles.append(rectangles[i])
            final_scores.append(scores[i])
            final_tags.append(tags[i])

    end_time = time.time()
    seconds = end_time - start_time
    G_Process_Stat['totalTime'] += seconds
    G_Process_Stat['totalCount'] += 1
    print(f'[process_frame_M]Elapsed time: {seconds:.2f} seconds, average time {G_Process_Stat["totalTime"] / G_Process_Stat["totalCount"]}, total inference count: {G_Process_Stat["totalCount"]}')

    return final_rectangles, final_scores, final_tags