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

import numpy as np
import scipy.ndimage as ndi
import threading as threading
import queue as queue
import SimpleITK as sitk
import numpy as np
from joblib import Parallel, delayed
import h5py as h5

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    return itkimage

def parse_itk_image(itk_image):
    numpyImage = sitk.GetArrayFromImage(itk_image)
    numpyOrigin = np.array(list(itk_image.GetOrigin()))
    numpySpacing = np.array(list(itk_image.GetSpacing()))
    numpyOrientation = np.array(list(itk_image.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyOrientation

def resample_itk_image(itk_image, target_size, target_spacing, padding_value):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(padding_value)
    return resampler.Execute(itk_image)

def normalizePlanes(npzarray):
    maxHU = 350.
    minHU = -1150.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def transform_matrix_offset_center(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=1, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

# run the image generator. Callable from joblib to parallelize the work
def _generate_image (i, j, X, y, image_data_generator, batch_x, dim_ordering,
                     depth_offset_min, depth_offset_max,
                     row_offset_min, row_offset_max, 
                     col_offset_min, col_offset_max):
    x = X[j]
    x = image_data_generator.random_transform(x.astype('float32'))
    
    if (dim_ordering == 'tf'):
        batch_x[i] = x[depth_offset_min:depth_offset_max,
                       row_offset_min:row_offset_max,
                       col_offset_min:col_offset_max,:]
    else:
        batch_x[i] = x[:,depth_offset_min:depth_offset_max,
                       row_offset_min:row_offset_max,
                       col_offset_min:col_offset_max]

# run the image generator. Callable from joblib to parallelize the work
def _generate_image_2 (i, j, X, y, file_map, index_map, image_data_generator, batch_x, dim_ordering,
                       depth_offset_min, depth_offset_max,
                       row_offset_min, row_offset_max, 
                       col_offset_min, col_offset_max,
                       num_output_channels, scalings, offsets):
    file_index = file_map[j]    
    x = X[file_index][j-index_map[file_index]]
    
    if (dim_ordering == 'tf'):
        x = x.reshape(tuple(list(x.shape) + [1]))
    else:
        x = x.reshape(tuple([1] + list(x.shape)))    
        
    x = image_data_generator.random_transform(x.astype('float32'))
    #x = image_data_generator.standardize(x)        
    
    if (num_output_channels>1):
        x = np.repeat(x, num_output_channels, axis=3)
        for k in range(num_output_channels):
            x[:,:,:,k]*=scalings[k]
            x[:,:,:,k]+=offsets[k]
    
    if (dim_ordering == 'tf'):
        batch_x[i] = x[depth_offset_min:depth_offset_max,
                       row_offset_min:row_offset_max,
                       col_offset_min:col_offset_max,:]
    else:
        batch_x[i] = x[:,depth_offset_min:depth_offset_max,
                       row_offset_min:row_offset_max,
                       col_offset_min:col_offset_max]

        
class ImageDataGenerator3D(object):
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 depth_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depth_flip=False,
                 windowing_scale_range=0.,
                 windowing_intercept_range=0.,
                 rescale=None,
                 dim_ordering='tf'):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.depth_index = 2
            self.row_index = 3
            self.col_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.depth_index = 1
            self.row_index = 2
            self.col_index = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)
        self.windowing_scale_range = windowing_scale_range
        self.windowing_intercept_range = windowing_intercept_range

    def flow(self, X, y=None, batch_size=32, balance=True, shuffle=True, seed=None, output_depth=-1, output_rows=-1, output_cols=-1):
        return NumpyArrayIterator3D(
            X, y, self,
            batch_size=batch_size, balance=balance, shuffle=shuffle, seed=seed,
            output_depth=output_depth, output_rows=output_rows, output_cols=output_cols,
            dim_ordering=self.dim_ordering)
    
    def flowList(self, X, Y, batch_size=32, balance=False, shuffle=True, seed=None, output_depth=-1, output_rows=-1, output_cols=-1,
                num_output_channels=1, scalings=None, offsets=None):
        return NumpyArrayIterator3DList(
            X, Y, self,
            batch_size=batch_size, balance=balance, shuffle=shuffle, seed=seed,
            output_depth=output_depth, output_rows=output_rows, output_cols=output_cols,
            num_output_channels=num_output_channels, scalings=scalings, offsets=offsets,
            dim_ordering=self.dim_ordering)  

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

        return x

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_depth_index = self.depth_index - 1
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            thetax = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            thetay = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
            thetaz = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            thetax = 0
            thetay = 0
            thetaz = 0
        rotation_matrix = np.array([[np.cos(thetay)*np.cos(thetaz), 
                                     -np.sin(thetax)*np.sin(thetay)*np.cos(thetaz)-np.cos(thetax)*np.sin(thetaz), 
                                     -np.cos(thetax)*np.sin(thetay)*np.cos(thetaz)+np.sin(thetax)*np.sin(thetaz), 
                                     0],
                                    [np.cos(thetay)*np.sin(thetaz), 
                                     -np.sin(thetax)*np.sin(thetay)*np.sin(thetaz)+np.cos(thetax)*np.cos(thetaz), 
                                     -np.cos(thetax)*np.sin(thetay)*np.sin(thetaz)-np.sin(thetax)*np.cos(thetaz), 
                                     0],
                                    [np.sin(thetay), 
                                     np.sin(thetax)*np.cos(thetay), 
                                     np.cos(thetax)*np.cos(thetay), 
                                     0],
                                    [0, 0, 0, 1]])
                
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0
            
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range) * x.shape[img_depth_index]
        else:
            tz = 0

        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]])
        
        if self.shear_range:
            shear_xy = np.random.uniform(-self.shear_range, self.shear_range)
            shear_yx = np.random.uniform(-self.shear_range, self.shear_range)
            shear_xz = np.random.uniform(-self.shear_range, self.shear_range)
            shear_zx = np.random.uniform(-self.shear_range, self.shear_range)
            shear_yz = np.random.uniform(-self.shear_range, self.shear_range)
            shear_zy = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear_xy = 0
            shear_yx = 0
            shear_xz = 0
            shear_zx = 0
            shear_yz = 0
            shear_zy = 0
        shear_matrix = np.array([[1, shear_xy, shear_xz, 0],
                                 [shear_yx, 1, shear_yz, 0],
                                 [shear_zx, shear_zy, 1, 0],
                                 [0, 0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])
        
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w, d = x.shape[img_row_index], x.shape[img_col_index], x.shape[img_depth_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w, d)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)
        
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        if self.depth_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_depth_index)

        if self.windowing_scale_range>0. or self.windowing_intercept_range>0.:            
            a = 1 + np.random.uniform(-self.windowing_scale_range, self.windowing_scale_range)
            b = np.random.uniform(-self.windowing_intercept_range, self.windowing_intercept_range)
            x = x * a + b
            x[x>1.]=1.
            x[x<0.]=0.
        
        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
    
            
class Iterator(object):

    def __init__(self, N, y, batch_size, balance, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, y, batch_size, balance, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, y, batch_size, balance=True, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        if (batch_size<0):
            batch_size = y.shape[0]
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                # balance input
                if (balance and y is not None):                    
                    neg_count = y.shape[0] - (y>0).sum()
                    max_label = y.max()
                    for i in range(1,max_label+1):
                        pos_i_count = (y==i).sum()                        
                        pos_i_to_add = neg_count - pos_i_count
                        if (pos_i_to_add>0 and pos_i_count>0):
                            pos_i_indices = np.arange(y.shape[0])[y==i]
                            if shuffle:
                                if seed is not None:
                                    np.random.seed(seed + self.total_batches_seen)
                                # shuffle positive not to add always the same
                                np.random.shuffle(pos_i_indices)
                            pos_i_added = 0
                            while (pos_i_added<pos_i_to_add):
                                if (pos_i_to_add-pos_i_added <= pos_i_indices.shape[0]):
                                    index_array = np.concatenate((pos_i_indices[:pos_i_to_add-pos_i_added], index_array))
                                    pos_i_added = pos_i_to_add #end loop
                                else:
                                    # append the whole set
                                    index_array = np.concatenate((pos_i_indices, index_array))
                                    pos_i_added += pos_i_indices.shape[0]                
                                    
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    np.random.shuffle(index_array)

            count = index_array.shape[0]
            current_index = (self.batch_index * batch_size)
            if (batch_size <= count ) and (count >= current_index + batch_size):
                self.batch_index += 1
                yield (index_array[current_index: current_index + batch_size],
                   current_index, batch_size)
            else:
                self.batch_index = 0
                current_index = 0             
                self.total_batches_seen += 1

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator3D(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size, balance=False, shuffle=False, seed=None,
                 output_depth=-1, output_rows=-1, output_cols=-1,
                 dim_ordering='tf'):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.output_depth = output_depth
        self.output_rows = output_rows
        self.output_cols = output_cols  
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.depth_index = 2
            self.row_index = 3
            self.col_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.depth_index = 1
            self.row_index = 2
            self.col_index = 3
        depth_count = self.X.shape[self.depth_index]
        row_count = self.X.shape[self.row_index]
        col_count = self.X.shape[self.col_index]
        self.depth_offset_min = 0
        self.depth_offset_max = depth_count
        self.row_offset_min = 0
        self.row_offset_max = row_count
        self.col_offset_min = 0
        self.col_offset_max = col_count
        if (self.output_depth > 0):
            depth_mid = depth_count // 2
            self.depth_offset_min = depth_mid - self.output_depth // 2
            self.depth_offset_max = depth_mid + self.output_depth // 2
            if (self.output_depth%2>0):
                self.depth_offset_max += 1
            depth_count = self.output_depth
        if (self.output_rows > 0):
            row_mid = row_count // 2
            self.row_offset_min = row_mid - self.output_rows // 2
            self.row_offset_max = row_mid + self.output_rows // 2
            if (self.output_rows%2>0):
                self.row_offset_max += 1
            row_count = self.output_rows
        if (self.output_cols > 0):
            col_mid = col_count // 2
            self.col_offset_min = col_mid - self.output_cols // 2
            self.col_offset_max = col_mid + self.output_cols // 2
            if (self.output_cols%2>0):
                self.col_offset_max += 1
            col_count = self.output_cols
        if (self.dim_ordering=='tf'):
            self.output_shape = (self.X.shape[0], depth_count, row_count, col_count, self.X.shape[4])
        else:
            self.output_shape = (self.X.shape[0], self.X.shape[1], depth_count, row_count, col_count)
        super(NumpyArrayIterator3D, self).__init__(self.X.shape[0], y, batch_size, balance, shuffle, seed)    
        
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel        
        batch_x = np.zeros(tuple([current_batch_size] + list(self.output_shape)[1:]), dtype=np.float32)                
        
        #run image generation in a multi-threaded parallel loop using all processors (n_jobs=-1, backend='threading')
        Parallel(n_jobs=-1, backend="threading")(delayed(_generate_image)(i, j, self.X, self.y, self.image_data_generator,
                                                                          batch_x, self.dim_ordering,
                                                                          self.depth_offset_min, self.depth_offset_max,
                                                                          self.row_offset_min, self.row_offset_max,
                                                                          self.col_offset_min, self.col_offset_max)
                                                for i, j in enumerate(index_array))        
        if self.y is None:
            return batch_x        
        batch_y = self.y[index_array]
        return batch_x, batch_y

class NumpyArrayIterator3DList(Iterator):

    def __init__(self, x_list, y_list, image_data_generator,
                 batch_size, balance=False, shuffle=False, seed=None,
                 output_depth=-1, output_rows=-1, output_cols=-1,
                 num_output_channels=1, scalings=None, offsets=None,
                 dim_ordering='tf'):
        if len(x_list)==0:
            raise Exception('Input list is empty')
        if (len(x_list)!=len(y_list)):
            raise Exception('lists size mismatch')
        if (num_output_channels>1):
            if (scalings is None or offsets is None):
                raise Exception('scalings / offsets arrays are None')
            if (scalings.shape[0]!=num_output_channels or offsets.shape[0]!=num_output_channels):
                raise Exception('scalings / offsets arrays not matching num_output_channels')
            
        self.x_list = x_list
        self.y_list = y_list
        self.image_data_generator = image_data_generator
        self.output_depth = output_depth
        self.output_rows = output_rows
        self.output_cols = output_cols  
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.depth_index = 2
            self.row_index = 3
            self.col_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.depth_index = 1
            self.row_index = 2
            self.col_index = 3
        depth_count = self.x_list[0].shape[self.depth_index]
        row_count = self.x_list[0].shape[self.row_index]
        col_count = self.x_list[0].shape[self.col_index]
        self.depth_offset_min = 0
        self.depth_offset_max = depth_count
        self.row_offset_min = 0
        self.row_offset_max = row_count
        self.col_offset_min = 0
        self.col_offset_max = col_count
        if (self.output_depth > 0):
            depth_mid = depth_count // 2
            self.depth_offset_min = depth_mid - self.output_depth // 2
            self.depth_offset_max = depth_mid + self.output_depth // 2
            depth_count = self.output_depth
        if (self.output_rows > 0):
            row_mid = row_count // 2
            self.row_offset_min = row_mid - self.output_rows // 2
            self.row_offset_max = row_mid + self.output_rows // 2
            row_count = self.output_rows
        if (self.output_cols > 0):
            col_mid = col_count // 2
            self.col_offset_min = col_mid - self.output_cols // 2
            self.col_offset_max = col_mid + self.output_cols // 2
            col_count = self.output_cols
            
        self.total_count = 0
        for i in range(len(self.x_list)):
            count = self.x_list[i].shape[0]
            self.total_count += count
            
        self.y = np.zeros(shape=(tuple([self.total_count])+tuple(self.y_list[0].shape[1:])), dtype=self.y_list[0].dtype)
        self.file_map = np.zeros(shape=(self.total_count), dtype=np.int32)
        self.index_map = np.zeros(shape=(len(self.x_list)), dtype=np.int32)
        current_index = 0        
        for i in range(len(self.y_list)):
            count = self.y_list[i].shape[0]
            self.y[current_index:current_index+count]=self.y_list[i]
            self.file_map[current_index:current_index+count] = i
            self.index_map[i] = current_index
            current_index += count
        if (num_output_channels<1):
            num_output_channels=1
        self.num_output_channels=num_output_channels
        self.scalings=scalings
        self.offsets=offsets
        if (self.dim_ordering=='tf'):
            self.output_shape = (self.total_count, depth_count, row_count, col_count, num_output_channels)
        else:
            self.output_shape = (self.total_count, num_output_channels, depth_count, row_count, col_count)
        super(NumpyArrayIterator3DList, self).__init__(self.total_count, self.y, batch_size, balance, shuffle, seed)    
        
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        # The transformation of images is not under thread lock so it can be done in parallel        
        batch_x = np.zeros(tuple([current_batch_size] + list(self.output_shape)[1:]), dtype=np.float32)
        
        #run image generation in a multi-threaded parallel loop using all processors (n_jobs=-1, backend='threading')
        Parallel(n_jobs=-1, backend="threading")(delayed(_generate_image_2)(i, j, self.x_list, self.y, 
                                                                            self.file_map, self.index_map,
                                                                            self.image_data_generator,
                                                                            batch_x, self.dim_ordering,
                                                                            self.depth_offset_min, self.depth_offset_max,
                                                                            self.row_offset_min, self.row_offset_max,
                                                                            self.col_offset_min, self.col_offset_max,
                                                                            self.num_output_channels, self.scalings, self.offsets)
                                                for i, j in enumerate(index_array))        
        batch_y = self.y[index_array]        
        return batch_x, batch_y    
    
    def count(self):
        return self.total_count
    
    def negative_count(self):
        return (self.y==0).sum()
    
    def positive_count(self):
        return (self.y>0).sum()  
                        
                        
def _produce(queue, iterator, num_iterations):
    for i in range(num_iterations):
        batch_x, batch_y = iterator.next()
        queue.put([batch_x, batch_y])
            
class QueuedIterator(object):
    def __init__(self, iterator, num_iterations):
        self.iterator = iterator
        self.num_iterations = num_iterations
        self.queue = queue.Queue(maxsize=2)            
            
    def get_queue(self):
        return self.queue
            
    def produce(self):
        t = threading.Thread(target=_produce, args=(self.queue, self.iterator, self.num_iterations))
        t.daemon = True
        t.start()        
    