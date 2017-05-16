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

from optparse import OptionParser
import os.path

os.environ["CUDA_VISIBLE_DEVICES"] =  "0"

from kaggle_utils import *
import xgboost as xgb

def parse_args():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input_csv",
                      help="input csv containing the list of series to predict label, with a header")
    parser.add_option("-d", "--directory",
                      dest="directory", default='',
                      help="directory where physical files are stored (in mhd format)")
    parser.add_option("--skip_lung_seg", action="store_true", dest="skip_lung_seg")
    parser.add_option("--skip_nodules", action="store_true", dest="skip_nodules")
    parser.add_option("--skip_nodule_charac", action="store_true", dest="skip_nodule_charac")
    parser.add_option("--skip_masses", action="store_true", dest="skip_masses")
    parser.add_option("--skip_mass_charac", action="store_true", dest="skip_mass_charac")    
    parser.add_option("--skip_emphysema", action="store_true", dest="skip_emphysema")    
    parser.add_option("--skip_calcification", action="store_true", dest="skip_calcification")    
    (options, args) = parser.parse_args()
    return options


def predict(inputs, input_feature_file, model_file, output_file):
    print('predicting')
    
    features = read_csv(input_feature_file)
    features = features[1:] # skip header
    
    # build feature dict
    feature_dict = {}
    feature_count = 0
    for i in range(len(features)):
        row = features[i]
        patient_id = row[0]
        feature_dict[patient_id] = row[1:-1] # remove last entry (label)
        if feature_count==0:
            feature_count = len(feature_dict[patient_id])
    
    X = np.zeros(shape=(len(inputs),feature_count), dtype=np.float32)
    for i in range(len(inputs)):
        row = inputs[i]
        patient_id = row[0]
        if patient_id not in feature_dict:
            print('WARNING: patient features are missing, prediction will defaul to 0.5', patient_id)
            continue
        f = feature_dict[patient_id]
        for j in range(feature_count):
            X[i,j] = float(f[j]) # convert str to float
    
    print('X shape',X.shape)
    
    # load model
    bst = xgb.Booster()
    bst.load_model(model_file) 
    
    xg_mat = xgb.DMatrix(X)
    Y=bst.predict(xg_mat)#, ntree_limit=7776)
    
    print('saving predictions to',output_file)
    
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    csvwriter.writerow(['id','cancer'])
    f.flush()
    
    for i in range(len(inputs)):
        row = inputs[i]
        patient_id = row[0]
        if patient_id not in feature_dict:
            # case has no feature: default prediction to 0.5
            csvwriter.writerow([inputs[i][0],str(0.5)])
        else:
            csvwriter.writerow([inputs[i][0],str(Y[i])])
        f.flush()
    f.close()

    print('predicting done.')


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
            print('WARNING: mhd file not found for series uid',seriesuid)
    
    # check final model existence
    final_model_file = 'train_results/final_model.bin'
    if not os.path.exists(final_model_file):
        print('ERROR: final model is missing (did you run the training first ?)')
        return
    
    output_directory = 'predict_results/'
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
    
    # predict
    output_prediction_file = output_directory + 'predictions.csv'
    predict(inputs, output_aggregated_file, final_model_file, output_prediction_file)

if __name__ == '__main__':
    main()
