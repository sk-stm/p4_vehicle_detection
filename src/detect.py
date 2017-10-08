import argparse
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label

from lesson_functions import *


class Detector(object):

    def __init__(self, svc, data_scaler, orient, pix_per_cell, cells_per_block, spatial_size, hist_bins, color_conv, cells_per_step, use_spatial_feat):
        self.svc = svc
        self.data_scaler = data_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.color_conv = color_conv
        self.cells_per_step = cells_per_step
        self.tracking = False
        self.use_spatial_feat = use_spatial_feat

        self.detection_state = {
            'car_ids': 0,
            'heat_map': np.array([0]),
            'detections': []
        }

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
            
                # Get spatial features
                spatial_features = np.array([]).reshape(-1)
                if self.use_spatial_feat:
                    bin_spatial(subimg, size=self.spatial_size)
                
                # Get histogram features
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.data_scaler.transform(test_features)  
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
        
        cv2.imshow("raw heatmap (normalized)", heatmap / np.max(heatmap))
        
        
        # merge in last detections
        if self.tracking and (self.detection_state['heat_map'].shape[0] != 0):
            heatmap = 0.6 * heatmap + 0.4 * self.detection_state['heat_map']
            cv2.imshow("merged heatmap (normalized)", heatmap / np.max(heatmap))

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

        cv2.imshow("thresholded heatmap (normalized)", heatmap / np.max(heatmap))

        # safe state
        if self.tracking:
            self.detection_state['heat_map'] = heatmap
        
        return filtered_detections

def process_image(img, scales, detector):
    # convert image data
    assert img.dtype == np.uint8 and np.max(img) > 1
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
    cv2.imshow("raw detections", debug_img)

    # non maximum suppression and filtering
    filtered_detections = detector.non_maximum_suppression(img, detections, threshold=2.2)
    logging.info("Got %i filtered detections.", len(filtered_detections))


    # assign detection IDs
    assigned_detections = []

    if not detector.tracking:
        assigned_detections = [(i, bb) for i, bb in enumerate(detections)]
    else:
        while len(filtered_detections):
            new_bb = filtered_detections.pop()

            match = False
            for i in range(len(detector.detection_state['detections'])):
                id, old_bb = detector.detection_state['detections'][i]
                # compute IoU
                iou = bb_intersection_over_union(new_bb, old_bb)
                if iou > 0.5:
                    match = True
                    assigned_detections.append((id, new_bb))
                    del detector.detection_state['detections'][i] 
                    break

            if not match:
                new_id = detector.detection_state['car_ids']
                detector.detection_state['car_ids'] += 1
                new_detection = (new_id, new_bb)
                assigned_detections.append(new_detection)
        
        # apply the merged detections
        detector.detection_state['detections'] = assigned_detections

    
    debug_img = draw_detections(img, assigned_detections)


    cv2.imshow("filtered detections", debug_img)

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
    use_spatial_feat =  dist_pickle["spatial_feat"]

    cells_per_step = 1
    scales = [(1.0, 400, 528), (1.5, 390, 560), (2.0, 384, 576), (3.0, 360, 700), (4.0, 380, 720)]
    input_file = args['input']
    assert type(input_file) is str
    is_video = input_file.endswith('mp4')

    # init detector
    detector = Detector(svc, data_scaler, orient, pix_per_cell, cells_per_block, spatial_size, hist_bins, color_conv, cells_per_step, use_spatial_feat)

    if is_video:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(input_file)
        clip = clip.set_fps(1.)
        #clip = clip.subclip(22., 30.)
        
        detector.tracking = True
        
        white_clip = clip.fl_image(lambda img : process_image(img, scales, detector))
        output_file = input_file[:-4] + "_out" + input_file[-4:]
        output_file = output_file.split("/")[-1]
        white_clip.write_videofile("output_images/" + output_file, audio=False)
    else:
        img = mpimg.imread('test_images/test1.jpg')
        process_image(img, scales, detector)


if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file', type=str, default="svc_pickle.p")
    parser.add_argument('--input', type=str, default="test_video.mp4")

    args = vars(parser.parse_args())
    main(args)
