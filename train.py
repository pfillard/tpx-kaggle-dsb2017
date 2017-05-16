# Copyright (C) 2017 Therapixel / Pierre Fillard (pfillard@therapixel.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

from optparse import OptionParser
import os.path

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

from kaggle_utils import *
import xgboost as xgb

def parse_args():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input_csv",
                      help="input csv containing the list of series and their labels, with a header")
    parser.add_option("-d", "--directory",
                      dest="directory", default='',
                      help="directory where physical files are stored (in mhd format)")
    parser.add_option("-t", "--model_type", action="store", dest="model_type", default=1,
                      help="type of model to train. Possible values: 1/2 (default: 1)")
    parser.add_option("--skip_lung_seg", action="store_true", dest="skip_lung_seg")
    parser.add_option("--skip_nodules", action="store_true", dest="skip_nodules")
    parser.add_option("--skip_nodule_charac", action="store_true", dest="skip_nodule_charac")
    parser.add_option("--skip_masses", action="store_true", dest="skip_masses")
    parser.add_option("--skip_mass_charac", action="store_true", dest="skip_mass_charac")    
    parser.add_option("--skip_emphysema", action="store_true", dest="skip_emphysema")    
    parser.add_option("--skip_calcification", action="store_true", dest="skip_calcification")    
    (options, args) = parser.parse_args()
    return options    
    
def train_xgboost(input_file, model_type, output_file):
    data = read_csv(input_file)
    header = data[0]
    data = data[1:]
    feat_count = len(data[0])-2 # minus seriesuid and label
    X = np.zeros(shape=(len(data),feat_count), dtype=np.float32)
    Y = np.zeros(shape=(len(data)), dtype=np.float32)
    for i in range(len(data)):
        d = data[i]
        X[i] = d[1:-1]
        Y[i] = float(d[-1]) # label is last feature
        
    # optimal parameters - do not alter
    if model_type==1:
        # model 1: more balanced
        params = {
            'eta': 0.001, 
            'seed':0, 
            'subsample': 1, 
            'colsample_bytree': 1, 
            'colsample_bylevel': 1,
            'objective': 'binary:logistic', 
            'max_depth':4,
            'min_child_weight':5,
            'gamma':0.009,
            'max_delta_step':1,
            'eval_metric': 'logloss',
            'lambda':0.007,
            'alpha':0
        }
        num_iterations = 5950
    elif model_type==2:
        # model 2: biased towards test set ?
        params = {
            'eta': 0.01, 
            'seed':0, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'colsample_bylevel': 0.8,
            'objective': 'binary:logistic', 
            'max_depth':1,
            'min_child_weight':11,
            'gamma':0.001,
            'max_delta_step':1,
            'eval_metric': 'logloss',
            'lambda':2.5,
            'alpha':0
        } 
        num_iterations = 5760
    else:
        print('ERROR: model type is not recognized:',model_type)
        return

    xg_train = xgb.DMatrix(X, label=Y) # Create our DMatrix to make XGBoost more efficient

    gbt = xgb.train(params, xg_train, num_boost_round = num_iterations)
    print('saving model to', output_file)
    gbt.save_model(output_file)
    
    
def main():
    opts = parse_args()
    
    # check consistency of inputs
    inputs = read_csv(opts.input_csv)
    inputs = inputs[1:] # skip header
    
    seriesuids = []
    labels = []
    for i in range(len(inputs)):
        seriesuid = inputs[i][0]
        label = inputs[i][1]
        seriesuids.append(seriesuid)
        labels.append(label)
        if not os.path.exists(opts.directory + '/' + seriesuid + '.mhd'):
            print('warning: mhd file not found for series uid',seriesuid)
    
    output_directory = 'train_results/'
    if not os.path.exists(output_directory):
        print('creating result directory:',output_directory)
        os.makedirs(output_directory)
    
    # pre-processing: segment lungs
    lung_seg_directory = output_directory + 'lung_segmentations/'
    if not os.path.exists(lung_seg_directory):
        print('creating lung segmentation directory:', lung_seg_directory)
        os.makedirs(lung_seg_directory)
        
    # TODO: lung segmentation
    if not opts.skip_lung_seg:
        segment_lungs(seriesuids, opts.directory, lung_seg_directory)
        
    # nodule screening
    nodule_file = output_directory + 'output_nodules.csv'    
    if not opts.skip_nodules:
        screen_nodules (seriesuids, opts.directory, nodule_file)
    
    # nodule characterization
    nodule_characteristics_file = output_directory + 'output_nodules_characteristics.csv'
    if not opts.skip_nodule_charac:
        characterize_nodules(nodule_file, opts.directory, lung_seg_directory, nodule_characteristics_file)
    
    # mass screening
    masses_file = output_directory + 'output_masses.csv'
    if not opts.skip_masses:
        screen_masses(seriesuids, opts.directory, masses_file)
    
    # characterizing masses (only size matters)
    mass_characteristics_file = output_directory + 'output_masses_characteristics.csv'
    if not opts.skip_mass_charac:
        characterize_masses(masses_file, opts.directory, mass_characteristics_file)
    
    # scan emphyseme
    emphysema_hist_file = output_directory + 'output_emphysema_histogram.csv'
    if not opts.skip_emphysema:
        screen_emphysema_histogram(seriesuids, opts.directory, emphysema_hist_file)
    
    # screen aort calcifications
    aort_calci_hist_file = output_directory + 'output_aort_calci_histogram.csv'
    if not opts.skip_calcification:
        screen_aort_calcification_histogram(seriesuids, opts.directory, aort_calci_hist_file)
    
    # aggregate results
    output_aggregated_file = output_directory + 'output_aggregated.csv'
    aggregate_results(inputs, nodule_characteristics_file, mass_characteristics_file, 
                      emphysema_hist_file, aort_calci_hist_file, output_aggregated_file)
    
    # train XGboost
    output_model_file = output_directory + 'final_model%s.bin'%(opts.model_type)
    train_xgboost(output_aggregated_file, int(opts.model_type), output_model_file)
    
    
if __name__ == '__main__':
    main()