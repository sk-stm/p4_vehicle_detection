## Running the training
1. Set you `cwd` to the root directory of this repo. 
2. Download the training data (for out-of-the-box training to `"../train_data/vehicles/`and `"../train_data/vehicles/`)
3. run `python src/train.py` with the right parameters (see `python src/train.py --help`)

## Running the detection
1. Run `python src/detect.py` with the right parameters (see `python src/detect.py --help`) - either you use the pickle-files of my classifier (`svc_pickle.p`) or use your own new trained one
2. Lay back and enjoy the fancy tracking