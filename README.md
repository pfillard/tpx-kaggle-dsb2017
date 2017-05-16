Copyright (C) 2017 Therapixel / Pierre Fillard (pfillard@therapixel.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


### Kaggle - Data Science Bowl 2017 - Therapixel solution


*** RECOMMENDED / TESTED ENVIRONMENT ***
- Linux Ubuntu 16.04 x64
- 128GB or RAM
- 2TB of free disk space
- 2xIntel i7 CPUs
- 4xNVIDIA Titan X (Pascal) or NVIDIA P6000



*** SOFTWARE DEPENDENCIES ***
- python3.5 (from distribution)
- cuda 8
- cudnn 5.1
- tensorflow 1.0 patched with activation function "relu1" (see github repo https://github.com/pfillard/tensorflow/tree/r1.0_relu1)
- xgboost (from pip3)
- numpy (from pip3)
- scipy (from pip3)
- skimage (from pip3)
- SimpleITK (from pip3)
- h5py (from pip3)
- optparse (from pip3)
- probably others (all from pip3)



*** DATA DEPENDENCIES ***
- kaggle's data science bowl 2017 data: https://www.kaggle.com/c/data-science-bowl-2017/data


*** INSTRUCTIONS FOR USE ***

The source layout is as follows:
- python3 scripts at the root (train.py, predict.py, etc.)
- a `models/` folder containing pre-trained models for feature extraction (nodule extraction, etc.)
- a `feature_models/` folder containing scripts to re-train the feature extraction models (python notebooks)


Training the model and making predictions consists in 2 steps:
 1. feature extraction and model training -> `train.py`
 2. making predictions -> `predict.py`

Those 2 steps are detailed below.

Important: data are expected to be converted in ITK MetaImage (.mhd, .raw) format, a volumetric format. 
Some resources to convert DICOM to ITK MetaImage are listed below:
- https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html
- http://manpages.ubuntu.com/manpages/precise/man1/gdcm2vtk.1.html

Finally, training models for feature extraction is a time and resource-consuming process. Instructions can be found in the `feature_models` directory.



*** STEP 1 - MODEL TRAINING ***

Typing `python3 -m train -h` outputs:

    Usage: train.py [options]
    
    Options:
      -h, --help            show this help message and exit
      -i INPUT_CSV, --input=INPUT_CSV
                            input csv containing the list of series and their
                            labels, with a header
      -d DIRECTORY, --directory=DIRECTORY
                            directory where physical files are stored (in mhd
                            format)
      -t MODEL_TYPE, --model_type=MODEL_TYPE
                            type of model to train. Possible values: 1/2 (default: 1)
      --skip_lung_seg
      --skip_nodules
      --skip_nodule_charac
      --skip_masses
      --skip_mass_charac
      --skip_emphysema
      --skip_calcification


Typical usage:

    python3 -m train -i stage1_labels.csv -d /path/to/converted/dicoms
    
    
This command launches the model training from data defined in inputed csv files (with labels), and from a directory containing the DICOM converted at the previous step. It will perform sequentially:
- lung segmentation for each data
- nodule extraction and characterization
- mass extraction and characterization
- emhysema estimation
- degree of calcification estimation
then the training of the model will occure.

The results at each step are saved into the `train_results/` folder. Each step can be skipped using the
corresponding `--skip_xxx` command to ease the retry on failure.

This process may take quite some time depending on the ressources available (likely to be several days to process the entire database).

The final model is saved as `train_results/final_model.bin` and is the only file needed to predict new data.

The model type can be either 1 or 2. For each type, a different set of parameters is used to train the model, as 2 submissions were allowed. By default (if the `--model_type` argument is missing), type 1 is used. Both types need to be run to reproduce the best 2 submissions.



*** STEP 2 - PREDICTION ***

Typing `python3 -m predict -h` outputs:

    Usage: predict.py [options]

    Options:
      -h, --help            show this help message and exit
      -i INPUT_CSV, --input=INPUT_CSV
                            input csv containing the list of series to predict
                            label, with a header
      -d DIRECTORY, --directory=DIRECTORY
                            directory where physical files are stored (in mhd
                            format)
      --skip_lung_seg
      --skip_nodules
      --skip_nodule_charac
      --skip_masses
      --skip_mass_charac
      --skip_emphysema
      --skip_calcification


Typical usage:

    python3 -m predict -i stage1_sample_submission.csv -d /path/to/converted/dicoms
    
    
This command will predict the cancer probability for each of the entry listed in the input csv file. Predictions are written in file `predit_results/predictions.csv`.

Likewise the training step, this command will first extract features for each entry, and then perform prediction based on the model trained at the previous step.
