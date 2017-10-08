import argparse
import logging
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label

from lesson_functions import *


class Detector(object):

    def __init__(self, svc, data_scaler, orient, pix_per_cell, cells_per_block, spatial_size, hist_bins, color_conv, cells_per_step):
        self.svc = svc
        self.data_scaler = data_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.color_conv = color_conv
        self.cells_per_step = cells_per_step

    def find_cars(self, img, scale, y_start, y_end):
        detections = []
        
        img_tosearch = img[y_start:y_end,:,:].copy()
        ctrans_tosearch = convert_color(img_tosearch, conv=self.color_conv)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cells_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cells_per_block + 1 
        nfeat_per_block = self.orient*self.cells_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cells_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // self.cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // self.cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cells_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cells_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cells_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*self.cells_per_step
                xpos = xb*self.cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.data_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    detection = ((xbox_left, ytop_draw+y_start), (xbox_left+win_draw,ytop_draw+win_draw+y_start))
                    detections.append(detection)

        return detections

    def non_maximum_suppression(self, img, detections, threshold=2):
        heatmap = np.zeros_like(img[:,:,0], dtype=np.float)
        filtered_detections = []

        # Iterate through list of bboxes
        for box in detections:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
       

        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            filtered_detections.append(bbox)

        cv2.imshow("heatmap (thresholded)", heatmap / np.max(heatmap))
        return filtered_detections

def process_image(img, scales, detector):
    # convert image data
    img = img.astype(np.float32)/255
    assert img.dtype == np.float32 and np.max(img) <= 1.0

    # run multi-scale detection
    detections = []
    for (scale, y_start, y_end) in scales:
        scale_detections = detector.find_cars(img, scale, y_start, y_end)
        logging.info("Got %i detections at scale %.3f.", len(scale_detections), scale)
        detections += scale_detections
    logging.info("Got %i raw detections.", len(detections))
    
    debug_img = draw_boxes(img, detections)
    cv2.imshow("raw_detections", debug_img)

    # non maximum suppression and filtering
    filtered_detections = detector.non_maximum_suppression(img, detections, 1)
    logging.info("Got %i filtered detections.", len(filtered_detections))
    debug_img = draw_boxes(img, filtered_detections)
    cv2.imshow("filtered_detections", debug_img)

    cv2.waitKey(1)

    return (debug_img * 255).astype(np.uint8)


def main(args):
    # load feature parameters and pre-trained classifier from file
    pickle_file = args["pickle_file"]
    logging.info("Loading data from '%s'", pickle_file)
    dist_pickle = pickle.load( open(pickle_file, "rb" ) )
    svc = dist_pickle["svc"]
    data_scaler = dist_pickle["scaler"]
    orient = dist_pickle["hog_orient"]
    pix_per_cell = dist_pickle["hog_pix_per_cell"]
    cells_per_block = dist_pickle["hog_cells_per_block"]
    color_conv = dist_pickle["color_conv"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    cells_per_step = 1

    # init detector
    detector = Detector(svc, data_scaler, orient, pix_per_cell, cells_per_block, spatial_size, hist_bins, color_conv, cells_per_step)


    from moviepy.editor import VideoFileClip
    clip = VideoFileClip("project_video.mp4")
    #img = mpimg.imread('test_images/test1.jpg')

    #scales = [(0.5, 400, 464), (0.75, 400, 496), (1.0, 400, 528), (2.0, 384, 576), (3.0, 360, 700), (4.0, 380, 720)]
    scales = [(1.0, 400, 528), (2.0, 384, 576), (3.0, 360, 700), (4.0, 380, 720)]
    white_clip = clip.fl_image(lambda img : process_image(img, scales, detector))
    white_clip.write_videofile("project_video_out.mp4", audio=False)
    
    #process_image(img, scales, detector)
        



if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file', type=str, default="svc_pickle.p")

    args = vars(parser.parse_args())
    main(args)