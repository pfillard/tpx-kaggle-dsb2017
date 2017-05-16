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
import csv
import lidc as lidc
import numpy as np
import SimpleITK as sitk
import xgboost as xgb
from scipy import ndimage



def read_csv(filename):
    lines = []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines



def segment_lungs(seriesuids, data_directory, output_directory):
    print('segmenting lungs')
    
    models = ['models/exp_lung_segmentation_1_3/best_model_loss']
    target_spacing = [2.,2.,2.]

    for seriesuid in seriesuids:
        # find to which subset it corresponds
        series_filename = data_directory + '/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):
            print('series not found:', seriesuid)
            continue
        
        output_file = output_directory + '/' + seriesuid + '.mhd'
        if (os.path.isfile(output_file)):        
            print('segmentation already exists:', seriesuid)
            continue
    
        itk_image = lidc.load_itk_image(series_filename)
        volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)    
        padding_value = volume.min()
        img_z_orig, img_y_orig, img_x_orig = volume.shape
        img_z_new = int(np.round(img_z_orig*spacing[2]/target_spacing[2]))
        img_y_new = int(np.round(img_y_orig*spacing[1]/target_spacing[1]))
        img_x_new = int(np.round(img_x_orig*spacing[0]/target_spacing[0]))
        itk_image = lidc.resample_itk_image(itk_image, [img_x_new,img_y_new,img_z_new], target_spacing, int(padding_value))
        volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)    
        volume = volume.astype(np.float32)
        volume = lidc.normalizePlanes(volume)
    
        #_, score_map_1 = lidc.screen_volume_emphyseme(volume, models, min_candidates=-1, map_index=1)
        #_, score_map_2 = lidc.screen_volume_emphyseme(volume, models, min_candidates=-1, map_index=2)

        #lung_mask = (score_map_1+score_map_2)>0.5
        ## remove smaller blobs using morphological opening    
        ## lung_mask = ndimage.morphology.binary_opening(lung_mask, structure=se_filter)    
        #lung_mask = lung_mask.astype(dtype=np.int16)
    
        _, score_map = lidc.screen_volume_lung_segmentation(volume, models, min_candidates=-1)
        lung_mask = score_map>0.5
        lung_mask = lung_mask.astype(dtype=np.int16)
        
        # keep only left-right largest connected components
        labels, nb_labels = ndimage.label(lung_mask)
            
        mk,mj,mi = lung_mask.shape
        mid_i=mi//2
        l_left = labels[:,:,:mid_i]
        l_right = labels[:,:,mid_i:]

        # left
        max_label_l=-1
        max_count_l = 0
        for l in range(1,nb_labels+1):
            count = (l_left==l).sum()
            if count > max_count_l:            
                max_label_l = l
                max_count_l = count

        # right
        max_label_r=-1
        max_count_r = 0
        for l in range(1,nb_labels+1):
            count = (l_right==l).sum()
            if count > max_count_r:
                max_label_r = l
                max_count_r = count

        lung_mask[:]=0
        if max_label_l>0:
            lung_mask[labels==max_label_l]=1
        if max_label_r>0:
            lung_mask[labels==max_label_r]=1
    
        # save result
        itk_image = sitk.GetImageFromArray(lung_mask)
        itk_image.SetOrigin(origin)
        itk_image.SetSpacing(spacing)
        itk_image.SetDirection(orientation)            
        sitk.WriteImage(itk_image, output_file)
    
    print('segmenting lungs done.')

    
    
def screen_nodules(seriesuids, data_directory, output_file):
    print('screening nodules')
    
    models = ['models/exp_27_multipass_1/best_model_loss',
              'models/exp_27_multipass_2/best_model_loss',
              'models/exp_27_multipass_3/best_model_loss',
              'models/exp_27_multipass_4/best_model_loss',
              'models/exp_27_multipass_5/best_model_loss'
              ]
    
    print('saving result to:', output_file)
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    csvwriter.writerow(['seriesuid','coordX','coordY','coordZ','probability'])
    f.flush()
    
    index = 0
    for seriesuid in seriesuids:
        print('processing series %d/%d'%(index+1,len(seriesuids)))
        index+=1
        # look up for sub-folders containing resampled data
        series_filename = data_directory + '/0.625/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):
            # if not found, fall back to default folder
            series_filename = data_directory + '/' + seriesuid + '.mhd'
            if (not os.path.isfile(series_filename)):        
                print('series not found:', seriesuid)
                continue
            
        w_coords, i_coords, score_map = lidc.screen_itk_volume(series_filename, models, do_normalize=True, min_candidates=-1)
        
        for i in range(len(w_coords)):
            wc = w_coords[i]
            ic = i_coords[i]
            csvwriter.writerow([seriesuid,wc[0],wc[1],wc[2],score_map[ic[0],ic[1],ic[2]]])
        f.flush()
    f.close()
    print('screening nodules done.')

    
    
def characterize_nodules(nodule_file, data_directory, seg_directory, output_file):
    print('characterizing nodules')
    model_1 = 'models/exp_malignancy_size_regression_1/best_model_loss'
    model_2 = 'models/exp_malignancy_size_regression_2_sgd/best_model_loss'

    num_features = 256 # number of activation maps of the last layer
    
    nodules = read_csv(nodule_file)
    nodules = nodules[1:] # skip header
    nodule_dict = {}
    for i in range(len(nodules)):
        row = nodules[i]
        seriesuid = row[0]
        x, y, z, confidence = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        if seriesuid not in nodule_dict:
            nodule_dict[seriesuid] = []
        nodule_dict[seriesuid].append([x,y,z,confidence])
    
    print('saving result to:', output_file)
    
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    header = ['seriesuid','coordX','coordY','coordZ','probability','malignancy','size']
    for i in range(num_features):
        header.append('f%.2d'%i)
    csvwriter.writerow(header)
    f.flush()

    index = 0
    series_count = len(nodule_dict.keys())
    for seriesuid in nodule_dict:
        print('processing series %d/%d'%(index+1,series_count))
        index += 1
        
        # look up for sub-folders containing resampled data
        series_filename = data_directory + '/0.625/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):
            # if not found, fall back to default folder
            series_filename = data_directory + '/' + seriesuid + '.mhd'
            if (not os.path.isfile(series_filename)):        
                print('series not found:', seriesuid)
                continue

        seg_file = seg_directory + seriesuid + '.mhd'
        if (not os.path.isfile(seg_file)):        
            print('lung segmentation file not found for series:', seriesuid)
            continue
    
        coords = nodule_dict[seriesuid]
    
        # read and resample
        scores_1, _ = lidc.estimate_scores_from_coordinates(coords, series_filename, num_outputs=2, model=model_1)
        scores_2, features = lidc.estimate_scores_from_coordinates(coords, series_filename, num_outputs=2, model=model_2)
        scores = (scores_1+scores_2)*0.5
    
        lung_positions, _ = lidc.estimate_lung_position(coords, seg_file)
        
        for i in range(len(coords)):
            if ~(lung_positions[i][3]):
                # drop nodules outside the lungs
                continue
            row = [seriesuid, coords[i][0], coords[i][1], coords[i][2], coords[i][3], scores[i,0], scores[i,1]]
            for j in range(features.shape[1]):
                row.append(features[i,j])
            row.append(lung_positions[i][0]) # ratio_k
            row.append(lung_positions[i][1]) # ratio_j
            row.append(lung_positions[i][2]) # ratio_i
            csvwriter.writerow(row)
        f.flush()
    f.close()
    print('characterizing nodules done.')
    
    
    
def screen_masses(seriesuids, data_directory, output_file):
    print('screening masses')
    
    models = ['models/exp_27_multipass_1/best_model_loss',
              'models/exp_27_multipass_2/best_model_loss',
              'models/exp_27_multipass_3/best_model_loss',
              'models/exp_27_multipass_4/best_model_loss',
              'models/exp_27_multipass_5/best_model_loss'
              ]
    
    print('saving result to:', output_file)
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    csvwriter.writerow(['seriesuid','coordX','coordY','coordZ','probability'])
    f.flush()
    
    index = 0
    for seriesuid in seriesuids:
        print('processing series %d/%d'%(index+1,len(seriesuids)))
        index+=1
        series_filename = data_directory + '/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):        
            print('series not found:', seriesuid)
            continue
            
        w_coords, i_coords, score_map = lidc.screen_itk_volume(series_filename, models, do_normalize=True, 
                                                               min_candidates=-1, target_spacing=[3.,3.,3.])
        
        for i in range(len(w_coords)):
            wc = w_coords[i]
            ic = i_coords[i]
            csvwriter.writerow([seriesuid,wc[0],wc[1],wc[2],score_map[ic[0],ic[1],ic[2]]])
        f.flush()
    f.close()
    
    print('screening masses done.')
    
    
    
def characterize_masses(mass_file, data_directory, output_file):
    print('characterizing masses')
    
    model_1 = 'models/exp_malignancy_size_regression_1/best_model_loss'
    model_2 = 'models/exp_malignancy_size_regression_2_sgd/best_model_loss'
    
    masses = read_csv(mass_file)
    masses = masses[1:] # skip header
    mass_dict = {}
    for i in range(len(masses)):
        row = masses[i]
        seriesuid = row[0]
        x, y, z, confidence = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        if seriesuid not in mass_dict:
            mass_dict[seriesuid] = []
        mass_dict[seriesuid].append([x,y,z,confidence])
    
    print('saving result to:', output_file)
    
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    header = ['seriesuid','coordX','coordY','coordZ','probability','malignancy','size']
    csvwriter.writerow(header)
    f.flush()

    index = 0
    series_count = len(mass_dict.keys())
    for seriesuid in mass_dict:
        print('processing series %d/%d'%(index+1,series_count))
        index += 1
        
        series_filename = data_directory + '/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):        
            print('series not found:', seriesuid)
            continue
    
        coords = mass_dict[seriesuid]
    
        # read and resample
        scores_1, _ = lidc.estimate_scores_from_coordinates(coords, series_filename, num_outputs=2, model=model_1,
                                                            target_spacing=[3., 3., 3.])
        scores_2, features = lidc.estimate_scores_from_coordinates(coords, series_filename, num_outputs=2, model=model_2,
                                                                   target_spacing=[3., 3., 3.])
        scores = (scores_1+scores_2)*0.5
        
        for i in range(len(coords)):
            row = [seriesuid, coords[i][0], coords[i][1], coords[i][2], coords[i][3], scores[i,0], scores[i,1]]
            csvwriter.writerow(row)
        f.flush()
    f.close()
    
    print('characterizing masses done.')
    
    
    
def screen_emphysema_histogram(seriesuids, data_directory, output_file):
    print('screening emphysema')

    models = ['models/exp_emphyseme_2/best_model_loss']
    target_spacing = [0.625, 0.625, 0.625]

    print('saving result to:', output_file)
    
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    csvwriter.writerow(['seriesuid','eh0','eh1','eh2','eh3','eh4','eh5','eh6','eh7','eh8','eh9'])
    f.flush()
    
    index = 0
    for seriesuid in seriesuids:
        print('processing series %d/%d'%(index+1,len(seriesuids)))
        index+=1

        # look up for sub-folders containing resampled data
        series_filename = data_directory + '/0.625/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):
            # if not found, fall back to default folder
            series_filename = data_directory + '/' + seriesuid + '.mhd'
            if (not os.path.isfile(series_filename)):        
                print('series not found:', seriesuid)
                continue
        
        itk_image = lidc.load_itk_image(series_filename)
        volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)    
        
        if ~(spacing==target_spacing).all():
            # resample using itk
            print('resampling itk volume')
            padding_value = volume.min()
            img_z_orig, img_y_orig, img_x_orig = volume.shape
            img_z_new = int(np.round(img_z_orig*spacing[2]/target_spacing[2]))
            img_y_new = int(np.round(img_y_orig*spacing[1]/target_spacing[1]))
            img_x_new = int(np.round(img_x_orig*spacing[0]/target_spacing[0]))
            itk_image = lidc.resample_itk_image(itk_image, [img_x_new,img_y_new,img_z_new], target_spacing, int(padding_value))
            volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)
            
        volume = volume.astype(np.float32)
        volume = lidc.normalizePlanes(volume)
    
        _, score_map_1 = lidc.screen_volume_emphyseme(volume, models, min_candidates=-1, map_index=1)
        _, score_map_2 = lidc.screen_volume_emphyseme(volume, models, min_candidates=-1, map_index=2)

        lung_mask = (score_map_1+score_map_2)>0.5
        total_lung_count = (lung_mask>0).sum()
    
        emphyseme_hist_struct = np.histogram(score_map_2*lung_mask, 10, range=(1e-15,1.0))
        emphyseme_hist = emphyseme_hist_struct[0].astype(np.float32)
        emphyseme_hist /= total_lung_count
    
        csvwriter.writerow(tuple([seriesuid])+ tuple(emphyseme_hist))
        f.flush()    
    f.close()
    
    print('screening emphysema done.')
    
    
def screen_aort_calcification_histogram(seriesuids, data_directory, output_file):
    print('screening aort calcification')
    
    models = ['models/exp_calcification_1_2/best_model_loss']
    target_spacing = [2.0,2.0,2.0]
    patch_size = 64
    offset = patch_size//2
    
    print('saving result to:', output_file)
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    csvwriter.writerow(['seriesuid','ah01','ah02','ah03','ah04','ah05','ah06','ah07','ah08','ah09','ah10'])
    f.flush()
    
    index = 0
    for seriesuid in seriesuids:
        print('processing series %d/%d'%(index+1,len(seriesuids)))
        index+=1
        
        series_filename = data_directory + '/' + seriesuid + '.mhd'
        if (not os.path.isfile(series_filename)):        
            print('series not found:', seriesuid)
            continue

        itk_image = lidc.load_itk_image(series_filename)
        volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)    
        padding_value = volume.min()
        img_z_orig, img_y_orig, img_x_orig = volume.shape
        img_z_new = int(np.round(img_z_orig*spacing[2]/target_spacing[2]))
        img_y_new = int(np.round(img_y_orig*spacing[1]/target_spacing[1]))
        img_x_new = int(np.round(img_x_orig*spacing[0]/target_spacing[0]))
        itk_image = lidc.resample_itk_image(itk_image, [img_x_new,img_y_new,img_z_new], target_spacing, int(padding_value))
        volume, origin, spacing, orientation = lidc.parse_itk_image(itk_image)    
        volume = volume.astype(np.float32)
        volume_orig = volume.copy() # save a copy for later
        volume = lidc.normalizePlanes(volume)

        # reuse emphyseme network for aort detection. mad_index = 1 is aort (2 is heart)
        candidates, score_map = lidc.screen_volume_emphyseme(volume, models, min_candidates=1, map_index=1)

        # 1 == Aort
        if (len(candidates)==0):
            print('cannot find label 1 in series', seriesuid)
            csvwriter.writerow([seriesuid,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            f.flush()
            continue
                            
        c = []
        for i in range(len(candidates)):
            kk,jj,ii = candidates[i]
            c.append([kk,jj,ii,score_map[kk,jj,ii]])
        # sort by descending probability - keep max prob as aort candidate
        c = sorted(c, key=lambda vec:vec[3], reverse=True)
        
        kk,jj,ii,_ = c[0] # best aort candidate
        
        # scale original volume differently to compute calcification histogram
        maxHU = 1000.
        minHU = -1150.
        volume_orig = (volume_orig - minHU) / (maxHU - minHU)
        volume_orig[volume_orig>1] = 1.
        volume_orig[volume_orig<0] = 0.
        volume_orig = np.pad(volume_orig, ((offset,offset), (offset,offset), (offset,offset)), 
                             'constant', constant_values=((0, 0),(0, 0),(0, 0)))
                        
        patch = volume_orig[kk:kk+patch_size,jj:jj+patch_size,ii:ii+patch_size]
        
        hist = np.histogram(patch, 10, range=(0,1.0))
        hist = hist[0].astype(np.float32)
        hist /= patch_size*patch_size*patch_size
        
        csvwriter.writerow(tuple([seriesuid])+ tuple(hist))
        f.flush()
    f.close()
    
    print('screening aort calcification done.')
    
    
def aggregate_results(inputs, nodule_file, mass_file, emphysema_file, calci_hist_file, output_file):
    print('aggregating results')
    
    channels = 2 # retain worst N nodules
    
    nodule_data = read_csv(nodule_file)
    nodule_data = nodule_data[1:] # skip header
    mass_data = read_csv(mass_file)
    mass_data = mass_data[1:] # skip header
    emphysema_data = read_csv(emphysema_file)
    emphysema_data = emphysema_data[1:] # skip header
    calci_hist_data = read_csv(calci_hist_file)
    calci_hist_data = calci_hist_data[1:] # skip header
    
    nodule_feature_count = 0
    nodule_dict = {}
    for i in range(len(nodule_data)):
        nodule_struct = nodule_data[i]
        seriesuid = nodule_struct[0]
        if (float(nodule_struct[4])<=0.5): # threshold on nodule probability
            continue
        if seriesuid not in nodule_dict:
            nodule_dict[seriesuid] = []
        row = []
        for j in range(4,len(nodule_struct)-2): # skip seriesuid, x, y ,z and lung_y, lung_x
            row.append(float(nodule_struct[j]))
        nodule_dict[seriesuid].append(row)
        if nodule_feature_count==0:
            nodule_feature_count = len(row)
    
    mass_dict = {}
    for i in range(len(mass_data)):
        mass_struct = mass_data[i]
        seriesuid = mass_struct[0]
        if seriesuid not in mass_dict:
            mass_dict[seriesuid] = []
        mass_dict[seriesuid].append(mass_struct[4:7]) # retain prob, mal and size
    
    emphysema_dict = {}
    for i in range(len(emphysema_data)):
        emphysema_struct = emphysema_data[i]
        seriesuid = emphysema_struct[0]
        emphysema_dict[seriesuid] = emphysema_struct[8:11] # retain last 3 hist values
    
    calci_hist_dict = {}
    for i in range(len(calci_hist_data)):
        struct = calci_hist_data[i]
        seriesuid = struct[0]
        calci_hist_dict[seriesuid] = struct[8:11] # retain last 3 hist values
    
    print('saving result to:', output_file)
    f = open(output_file, 'w')
    csvwriter = csv.writer(f, delimiter=',')
    # write header
    header = ['seriesuid','eh0','eh1','eh2']
    for j in range(channels):
        for i in range(nodule_feature_count):
            header.append('nf%d_%d'%(j,i))            
    header.append('max_nod_size')
    header.append('prob_max_nod')
    header.append('max_malignancy')
    header.append('prob_max_mal')
    header.append('ah0')
    header.append('ah1')
    header.append('ah2')
    header.append('label')
    csvwriter.writerow(header)
    f.flush()
    
    for i in range(len(inputs)):
        seriesuid = inputs[i][0]
                            
        if seriesuid not in emphysema_dict:
            print('ATTENTION: missing data for series', seriesuid, ': no emphysema histogram! skipping ...')
            continue
            
        if seriesuid not in calci_hist_dict:
            print('ATTENTION: missing data for series', seriesuid, ': no calcification histogram! skipping ...')
            continue
        
        label = float(inputs[i][1])
        row = [seriesuid]
            
        # append emphysema histogram
        emphysema_row = emphysema_dict[seriesuid]
        for j in range(len(emphysema_row)):
            row.append(str(emphysema_row[j]))
            
        max_malignancy = 0.
        prob_max_mal = 0.
        max_nod_size = 0.
        prob_max_nod = 0.
            
        # append nodule features
        nodule_features = np.zeros(shape=((nodule_feature_count-2)*channels), dtype=np.float32)
        if seriesuid in nodule_dict:            
            nodule_struct = nodule_dict[seriesuid]
            # sort struct by malignancy * nodule_prob
            nodule_struct = sorted(nodule_struct, key=lambda vec: vec[0]*vec[1], reverse=True)
            for j in range(min(len(nodule_struct),channels)):
                size = nodule_struct[j][2]
                mal = nodule_struct[j][1]
                prob = nodule_struct[j][0]
                if mal>max_malignancy:
                    max_malignancy = mal
                    prob_max_mal = prob
                if size>max_nod_size:
                    max_nod_size = size
                    prob_max_nod = nodule_struct[j][0]
                nodule_features[j*(nodule_feature_count-2)] = nodule_struct[j][0]
                for k in range(3, nodule_feature_count): # skip prob, malignancy and size                    
                    nodule_features[j*(nodule_feature_count-2)+k-2] = nodule_struct[j][k]
            for j in range((nodule_feature_count-2)*channels):
                row.append(str(nodule_features[j]))
        else:
            for j in range((nodule_feature_count-2)*channels):
                row.append(str(0.0))            

        # retain largest mass size (if any)
        if seriesuid in mass_dict:
            masses = mass_dict[seriesuid]
            for j in range(len(masses)):
                mass_size = float(masses[j][-1])
                mass_mal = float(masses[j][-2])
                mass_prob = float(masses[j][-3])
                if (mass_prob>0.9 and mass_size>max_nod_size):
                    max_nod_size = mass_size
                    prob_max_nod = mass_prob
                if (mass_mal>max_malignancy):
                    max_malignancy = mass_mal
                    prob_max_mal = mass_prob

        row.append(str(max_nod_size))        
        row.append(str(prob_max_nod))
        row.append(str(max_malignancy))
        row.append(str(prob_max_mal))
        
        # append aort histogram
        calci_hist_row = calci_hist_dict[seriesuid]
        for j in range(len(calci_hist_row)):
            row.append(str(calci_hist_row[j]))
        
        # finally, append label
        row.append(str(label))
        
        csvwriter.writerow(row)
        f.flush()        
    f.close()
    
    print('aggregating results done.')