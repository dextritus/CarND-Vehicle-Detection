import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from scipy.ndimage.measurements import label
from collections import deque

#process the movie
from moviepy.editor import VideoFileClip

from util_functions import *
from sklearn.model_selection import train_test_split
import pickle 			#for saving and reading data from files

car_images = glob.glob('./vehicles/**/*.png', recursive = True)
noncar_images = glob.glob('./non-vehicles/**/*.png', recursive = True)

################################################
# parameters that can be tweaked, use intuition
################################################

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
# y_start_stop = [300, None] # Min and max in y to search in slide_window()

################################################
# feature extraction
################################################

# car_features = extract_features(car_images, color_space=color_space, 
# 						spatial_size=spatial_size, hist_bins=hist_bins, 
# 						orient=orient, pix_per_cell=pix_per_cell, 
# 						cell_per_block=cell_per_block, 
# 						hog_channel=hog_channel, spatial_feat=spatial_feat, 
# 						hist_feat=hist_feat, hog_feat=hog_feat)

# notcar_features = extract_features(noncar_images, color_space=color_space, 
# 						spatial_size=spatial_size, hist_bins=hist_bins, 
# 						orient=orient, pix_per_cell=pix_per_cell, 
# 						cell_per_block=cell_per_block, 
# 						hog_channel=hog_channel, spatial_feat=spatial_feat, 
# 						hist_feat=hist_feat, hog_feat=hog_feat)

# # # #stack all features for scaling
# X = np.vstack((car_features, notcar_features)).astype(np.float64)
# # # # apply the scaler
# X_scaler = StandardScaler().fit(X) # could also use fit_transform
# X_scaled = X_scaler.transform(X)

# y = np.hstack((np.ones(len(car_images)), np.zeros(len(noncar_images))))

# # # split into training and test vectors
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(
# 	X_scaled, y, test_size=0.2, random_state=rand_state)

# # # train the classifier with the training data
# svc = LinearSVC()
# t1 = time.time()
# svc.fit(X_train, y_train)
# t2 = time.time()

# print("It takes ", round((t2-t1),2), 'seconds to train the SVC classifier')
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# ##############################
# # save to pickled data
# ##############################
# pickle_data = {}
# pickle_file = open('pickle_file', 'wb')
# pickle_data['X_scaled'] = X_scaled
# pickle_data['X_scaler'] = X_scaler
# pickle_data['clsf'] = svc
# pickle.dump(pickle_data, pickle_file)
# pickle_file.close()

##############################
# load pickled data
##############################

pickle_file = open('pickle_file', 'rb')
pickle_data = pickle.load(pickle_file)
X_scaled = pickle_data['X_scaled']
X_scaler = pickle_data['X_scaler']
svc = pickle_data['clsf']
pickle_file.close()


# Define a single function that can extract features using hog sub-sampling and make predictions

ystart = 390
ystop = 656
scale = 2

########################################
# pipeline for each image
########################################
hist_wdw = deque(maxlen = 8)

def hist_wdw_avg(hist_wdw):
	hist_sum = np.zeros_like(hist_wdw[0])
	for i in range(0, len(hist_wdw)):
		hist_sum = np.add(hist_sum, hist_wdw[i])

	return hist_sum/len(hist_wdw)

def pipeline(img):
	# test_img = mpimg.imread('./test_images/test5.jpg')
	heat = np.zeros((720,1280)).astype(np.float)
	s_range = np.linspace(0.8, 3, 5)
	out_img = np.copy(img)
	for scale in s_range:
		out_img, boxes = find_cars(img, out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		heat = add_heat(heat, boxes)

	hist_wdw.append(heat)
	hist_avg = hist_wdw_avg(hist_wdw)
	# print("hist_avg ", np.max(hist_avg))
	heat = apply_threshold(hist_avg, 2.75)
	heatmap = np.clip(heat, 0, 255)

	labels = label(heatmap)

	draw_img = draw_labeled_bboxes(np.copy(img), labels)

	return draw_img


#########################################
# video files
########################################

# fname = "project_video"
# output = fname + "_output_final_revised.mp4"
# input_file = VideoFileClip(fname + ".mp4")
# # cut_file = input_file.subclip(41, 43)
# processed_file = input_file.fl_image(pipeline)
# processed_file.write_videofile(output, audio = False)

##########################################
# save images for writeup
###########################################

# test_img = mpimg.imread(car_images[0])
# print(test_img[:,:,0].shape)
# converted = cv2.cvtColor(test_img, cv2.COLOR_RGB2YCrCb)

# features, hog_image_ch1 = get_hog_features(converted[:,:,0], orient, 
#                         pix_per_cell, cell_per_block, 
#                         vis=True, feature_vec=False)

# features, hog_image_ch2 = get_hog_features(converted[:,:,1], orient, 
#                         pix_per_cell, cell_per_block, 
#                         vis=True, feature_vec=False)

# plt.figure()
# plt.subplot(221)
# plt.imshow(test_img, origin = 'upper')
# plt.ylabel('original')
# plt.subplot(222)
# plt.imshow(converted, origin = 'upper')
# plt.ylabel('converted')
# plt.subplot(223)
# plt.imshow(hog_image_ch1, origin = 'upper')
# plt.ylabel('HOG, first channel')
# plt.subplot(224)
# plt.imshow(hog_image_ch2, origin = 'upper')
# plt.ylabel('HOG, second channel')
# plt.savefig('./output_images/ycrcbhog.png')
# plt.show()


# test_img = mpimg.imread('./test_images/test5.jpg')
# heat = np.zeros((720,1280)).astype(np.float)
# s_range = np.linspace(0.8, 3, 5)
# out_img = np.copy(test_img)
# for scale in s_range:
# 	out_img, boxes = find_cars(test_img, out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# 	heat = add_heat(heat, boxes)

# heat_orig = np.copy(heat)
# heat = apply_threshold(heat,5)
# heatmap = np.clip(heat, 0, 255)

# labels = label(heatmap)

# draw_img = draw_labeled_bboxes(np.copy(test_img), labels)

# plt.figure()
# plt.subplot(221)
# plt.imshow(out_img, origin = 'upper')
# plt.ylabel('sliding windows')
# plt.subplot(222)
# plt.imshow(heat_orig, origin = 'upper', cmap = 'hot')
# plt.ylabel('heat map')
# plt.subplot(223)
# plt.imshow(heatmap, origin = 'upper', cmap = 'hot')
# plt.ylabel('thresholded heatmat')
# plt.subplot(224)
# plt.imshow(draw_img, origin = 'upper')
# plt.ylabel('final detection')
# plt.savefig('./output_images/window.png')
# plt.show()

