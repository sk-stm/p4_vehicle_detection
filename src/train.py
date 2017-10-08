import argparse
import glob
import logging
import numpy as np
import os
import pickle
import random
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lesson_functions import *


def load_traindata(cars_dir, notcars_dir, subset, balance=True):
    # load image paths
    cars_dir = os.path.abspath(cars_dir)
    logging.debug("Loading vehicle images from %s ", cars_dir)
    cars = glob.glob(cars_dir, recursive=True)
    logging.debug("Loaded %i vehicle samples", len(cars))

    notcars_dir = os.path.abspath(notcars_dir)
    logging.debug("Loading non-vehicle images from %s ", notcars_dir)
    notcars = glob.glob(notcars_dir, recursive=True)
    logging.debug("Loaded %i non-vehicle samples", len(notcars))

    assert len(cars) > 1000 and len(notcars) > 1000

    # shuffe dataset
    for _ in range(3):
        random.shuffle(cars)
        random.shuffle(notcars)

    # balance dataset
    if balance:
        if len(cars) > len(notcars):
            cars = cars[:len(notcars)]
            logging.debug("Balancing: strapping cars dataset to %i samples", len(cars))
        elif len(notcars) > len(cars):
            notcars = notcars[:len(cars)]
            logging.debug("Balancing: strapping notcars dataset to %i samples", len(notcars))

    # check if only a subset is wanted
    if subset:
        logging.warn("Extracting only a subset of the input data!")
        cars = cars[:2000]
        notcars = notcars[:2000]

    return cars, notcars

def prepare_test_data(car_features, notcar_features):
    # stack data
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return scaled_X, y, X_scaler


def train_classifier(X, y, with_test=False, test_size=0.3):

    # train on subset
    if with_test:
        logging.debug("Training test SVC with %i%% of the data ..", (1.0-test_size)*100)
        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        SVC(probability=True, C=10., kernel='rbf')
        svc.fit(X_train, y_train)
        logging.debug('Test accuracy of SVC = %f', round(svc.score(X_test, y_test), 4))

    # now train on all training data
    logging.debug("Training with all the data ..")
    svc = SVC(probability=True, C=10., kernel='rbf')
    svc.fit(X, y)
    return svc


def main(args):
    # load the train images
    logging.info("Loading train images ..")
    cars, notcars = load_traindata(args['cars'], args['notcars'], args['subset'])

    # define the feature parameters
    color_conv = 'RGB2YCrCb'
    orient = 9  # HOG orientations
    pix_per_cell = 16 # HOG pixels per cell
    cells_per_block = 3 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = False # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    logging.info("HOG: using %i orientations, %i pixels per cell and %i cells per block", orient, pix_per_cell, cells_per_block)

    # extract features
    logging.info("Extracting train features ..")
    car_features = extract_features(cars, color_conv=color_conv, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cells_per_block=cells_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_conv=color_conv, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cells_per_block=cells_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    assert len(car_features) == len(cars) and len(notcar_features) == len(notcars)
    logging.info("Total feature vector lenght: %i", len(car_features[0]))

    # prepare train data
    logging.info("Scaling data to zero mean and unit variance ..")
    X, y, scaler = prepare_test_data(car_features, notcar_features)

     # train the classifier
    logging.info("Training the SVC ..")
    svc = train_classifier(X, y, args['with_test'])

    # store the data
    logging.info("Finished. Stroring data to '%s' ..", args['outfile'])
    storage = {}
    storage['svc'] = svc
    storage['scaler'] = scaler
    storage['hog_orient'] = orient
    storage['hog_pix_per_cell'] = pix_per_cell
    storage['hog_cells_per_block'] = cells_per_block
    storage['hog_channel'] = hog_channel
    storage['color_conv'] = color_conv
    storage['hist_bins'] = hist_bins
    storage['spatial_size'] = spatial_size
    storage['spatial_feat'] = spatial_feat
    storage['hist_feat'] = hist_feat
    storage['hog_feat'] = hog_feat

    pickle.dump(storage, open(args['outfile'], 'wb'))
    logging.info("done!")


if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cars', type=str, default="../train_data/vehicles/**/*.png")
    parser.add_argument('--notcars', type=str, default="../train_data/non-vehicles/**/*.png")
    parser.add_argument('--outfile', type=str, default="svc_pickle.p")
    parser.add_argument('--subset', type=bool, default=False)
    parser.add_argument('--with_test', type=bool, default=False)

    args = vars(parser.parse_args())
    main(args)
