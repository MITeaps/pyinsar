import re
from collections import OrderedDict
import h5py

import numpy as np
from scipy.constants import c
from tqdm import tqdm
from skimage.filters import threshold_otsu
import gdal

from pyinsar.processing.utilities import insar_simulator_utils
from pyinsar.processing.geography.coordinates import compute_x_and_y_coordinates_maps

def divide_into_squares(image, size, stride):
    """
    Create many patches from an image

    Will drop any patches that contain NaN's

    @param image: Source image
    @param size: Size of one side of the square patch
    @param stride: Spacing between patches (must be an integer greater than 0)
    @returns List containing the extent of each patch and a list of the patches
    """

    array_list = []
    extent_list = []

    for x in range(0, image.shape[-1]-size, stride):
        for y in range(0, image.shape[-2]-size, stride):
            if image.ndim == 2:
                cut_box = image[y:y+size, x:x+size]
            elif image.ndim == 3:
                cut_box = image[:, y:y+size, x:x+size]

            if not np.any(np.isnan(cut_box)):
                extent_list.append((x, x+size, y, y+size))
                array_list.append(cut_box)

    return extent_list, array_list


def generate_minimum_ground_range_limits(satellite_height, incidence_ranges, image_size):
    """
    Determine the limits of minimum ground ranges of a satellite pass

    @param satellite_height: Height of satellite
    @param incidence_ranges: Range of valid incidence angles (shape of [:,2])
    @param image_size: Length of image
    @return range of possible minimum ground ranges
    """

    minimum_ground_ranges = np.tan(incidence_ranges) * satellite_height
    minimum_ground_ranges[:,1] = minimum_ground_ranges[:,1] -  np.sqrt(2) * image_size

    return minimum_ground_ranges


def generate_phase_samples_from_looks_and_ranges(deformation_list, xx, yy, satellite_height, track_angles, minimum_ground_ranges,
                                                 size = (100,100), dtype=np.float32):
    """
    Generates different possible phases from a list of deformations due to different track angles and groud ranges.

    @param deformation_list: List of deformations
    @param xx: x coordinates
    @param yy: y coordinates
    @param satellite_height: Height of satellite
    @param track_angles: Iterable of track angles
    @param minimum_ground_ranges: Iterable of minimum ground ranges
    @param size: Tuple giving the size of each deformation in deformation_list
    @param dtype: Data type
    @return array containing paramaters and and array containing phases
    """

    radar_wavelength = c / 5.405000454334350e+09

    num_images = len(track_angles) * len(minimum_ground_ranges) * len(deformation_list)
    phases = np.zeros([num_images] + list(size), dtype=dtype)
    params = np.zeros([num_images, 2], np.int)

    index = 0

    for deformation in tqdm(deformation_list):
        for angle in track_angles:
            for ground_range in minimum_ground_ranges:
                phases[index, :, :] = insar_simulator_utils.generate_interferogram_from_deformation(angle, ground_range, satellite_height, True,
                                                                                                    radar_wavelength, 2, deformation, xx, yy, 0)
                params[index,0] = angle
                params[index,0] = ground_range

                index +=1

    return params, phases

def generate_phase_samples(deformation, satellite_height, radar_wavelength, cell_size, image_size, stride=20):
    """
    **In Development** Generate phase samples by tiling an array of deformations

    @param deformation: Array containing deformation
    @param satellite_height: Height of Satellite
    @param radar_wavelength: Wavelength of radar
    @param cell_size: Size of cell (length of one side of the cell)
    @param image_size: Ignored?
    @param stride: Distance between tiles
    """
    x_bound_start, x_bound_end, y_bound_start, y_bound_end = determine_x_y_bounds(deformation, x_coords, y_coords,offset=0,
                                                                                  threshold_function=threshold_otsu)

    x_img_start, x_img_end, y_img_start, y_img_end = insar_simulator_utils.determine_deformation_bounding_box(deformation,
                                                                                                              threshold_function=threshold_otsu)

    sim_extents, sim_patches = divide_into_squares(deformation[:, y_img_start: y_img_end+1, x_img_start:x_img_end+1], size, 20)

    track_angles = np.linspace(0,2*np.pi, endpoint=False, num=36)

    incidence_ranges = np.deg2rad(np.array([(29.16,36.59), (34.77, 41.85), (40.04,46)]))
    image_size = 100 * 100
    minimum_ground_ranges = np.tan(incidence_ranges) * satellite_height
    minimum_ground_ranges[:,1] = minimum_ground_ranges[:,1] -  np.sqrt(2) * image_size

    min_ground_range_array = np.array([], dtype=np.float)
    for i in range(3):
        min_ground_range_array = np.concatenate([min_ground_range_array, np.linspace(*minimum_ground_ranges[i,:], num=5)])


    start_x = -1*image_size*cell_size / 2
    end_x = image_size*cell_size / 2

    start_y = -1*image_size*cell_size / 2
    end_y = image_size*cell_size / 2

    cut_x_coords, cut_y_coords = compute_x_and_y_coordinates_maps(start_x,
                                                                  end_x,
                                                                  start_y,
                                                                  end_y,
                                                                  cell_size,
                                                                  cell_size)

    params, phases = generate_phase_samples_from_deformation(sim_patches, cut_x_coords, cut_y_coords, satellite_height,
                                                             track_angles,min_ground_range_array)

# def generate_index(data_file, label):
#     key_list = [ re.search('.*data$', key_name).group() for key_name in data_file.keys()
#                  if re.search('.*data$', key_name) is not None]
#     key_list.sort()

#     num_images = np.zeros(len(key_list), dtype=np.int)
#     index_dict = OrderedDict()

#     for index, key in enumerate(key_list):
#         index_dict[index] = key
#         num_images[index] = data_file[key].shape[0]

#     final_index = np.zeros((np.sum(num_images),3), dtype=int)
#     final_index[:,0] = label

#     current_index = 0
#     for index in range(len(num_images)):
#         tmp_num_images = num_images[index]
#         index_slice = slice(current_index, tmp_num_images+current_index)
#         final_index[index_slice, 1] = index
#         final_index[index_slice, 2] = np.arange(tmp_num_images)
#         current_index += tmp_num_images


#     return index_dict, final_index



# def retrieve_data(index, index_dict, data_file, size):
#     final_data = np.zeros([index.shape[0]] + list(size))

#     datasets = np.unique(index[:,0])

#     for i in range(len(datasets)):
#         dataset_name = index_dict[datasets[i]]

#         dataset_index = index[:,0] == datasets[i]

#         final_data[dataset_index,:,:] = data_file[dataset_name][index[dataset_index,1],:,:]

#     return final_data


class DataRetriever(object):
    """
    Class for retrieving data from an hdf file
    """
    def __init__(self, file_name_list, label_list, size, dtype=np.float32, chunk_size = 1000):
        """
        Initilaize DataRetriever object

        @param file_name_list: List of hdf filenames
        @param label_list: List of labels
        @param size: Tuple containing the size of the images
        @param dtype: Data type of images
        @param chunk_size: Size of chunks to use when reading data
        """
        self.label_list = label_list
        self.size = list(size)
        self.dtype = dtype
        self.chunk_size = 1000

        self.data_file_dict = OrderedDict()
        for file_name, label in zip(file_name_list, self.label_list):
            self.data_file_dict[label] = OrderedDict()
            self.data_file_dict[label]['data_file'] = h5py.File(file_name,'r')

            self.data_file_dict[label]['key_list'] = [ re.search('.*data$', key_name).group() for key_name
                                                       in self.data_file_dict[label]['data_file']
                                                       if re.search('.*data$', key_name) is not None ]


            self.data_file_dict[label]['key_list'].sort()

            self.data_file_dict[label]['image_index'] = []
            current_index = 0
            for i, key in enumerate(self.data_file_dict[label]['key_list']):
                num_images = self.data_file_dict[label]['data_file'][key].shape[0]
                self.data_file_dict[label]['image_index'].append(num_images + current_index)

                current_index += num_images

            self.data_file_dict[label]['image_index'] = np.array(self.data_file_dict[label]['image_index'])
            self.data_file_dict[label]['num_images'] = current_index

    def get_num_images(self):
        """
        Get the number of images for each label

        @return Number of images associated with each label
        """
        num_images = np.zeros((len(self.label_list), 2), dtype=np.int)

        num_images[:,0] = self.label_list

        for index, label in enumerate(self.label_list):
            num_images[index, 1] = self.data_file_dict[label]['num_images']

        return num_images


    def _retrieve_hdf_data(self, label, dataset_name, index):
        """
        Retrieve data from a given label

        @param label: Label of data
        @param dataset_name: Name of dataset
        @param index: Index of data to retrieve

        @return Requested images
        """
        sorted_index = np.argsort(index)

        if self.chunk_size is None or self.chunk_size == 0:
            return self.data_file_dict[label]['data_file'][dataset_name][index[sorted_index],:,:][sorted_index,:,:]

        else:
            data_shape = self.data_file_dict[label]['data_file'][dataset_name].shape
            dtype = self.data_file_dict[label]['data_file'][dataset_name].dtype

            return_data = np.zeros([len(index)] + list(data_shape[1:]), dtype = dtype)

            sorted_index_list = np.split(index[sorted_index], range(self.chunk_size, len(index), self.chunk_size))
            return_index = np.split(sorted_index, range(self.chunk_size, len(index), self.chunk_size))

            for index_chunk, return_index_chunk in zip(sorted_index_list, return_index):
                return_data[return_index_chunk] = self.data_file_dict[label]['data_file'][dataset_name][index_chunk,:,:]

            return return_data

    def _get_images_from_label(self, label, index):
        """
        Get images from a particular label

        @param label: Label of data to retrieve
        @param index: Index of data to retrieve

        @return Requested images
        """
        group_index = np.searchsorted(self.data_file_dict[label]['image_index'], index, side='right')

        num_images_before_index = np.roll(self.data_file_dict[label]['image_index'], shift=1)
        num_images_before_index[0] = 0

        local_index = index - num_images_before_index[group_index]

        final_data = np.zeros([len(index)] + self.size, dtype=self.dtype)

        datasets = np.unique(group_index)

        for i in range(len(datasets)):
            dataset_name = self.data_file_dict[label]['key_list'][datasets[i]]

            dataset_index = group_index == datasets[i]

            final_data[dataset_index,:,:] = self._retrieve_hdf_data(label, dataset_name, local_index[dataset_index])

        return final_data

    def get_images(self, index):
        """
        Retrieve images given by index

        @param index: Array with shape [:,2], first column label, second column index
        @return Requested images
        """
        valid_labels = np.unique(index[:,0])

        image_data = np.zeros([index.shape[0]] + self.size)

        for label in valid_labels:

            label_index = index[:,0] == label
            image_data[label_index,:,:] = self._get_images_from_label(label, index[label_index, 1])

        return image_data


def rotate_image_list(in_image_extents, in_image_list, progress=True):
    """
    Rotate input images 0, 90, 180, and 270 degrees

    @param in_image_extents: List of the extents of the images being rotated
    @param in_image_list: List of images to rotate
    @param progress: Show a progress bar

    @return array of image extents, and array of rotated images
    """


    new_image_data = np.zeros([len(in_image_extents)*4, 100, 100], dtype=in_image_list[0].dtype)
    new_image_extents = np.zeros([len(in_image_extents)*4, 4], dtype=np.int)

    index = 0
    for image, extents in tqdm(zip(in_image_list, in_image_extents), total=len(in_image_list), disable = not progress):
        for i in range(4):
            new_image_data[index,:,:] = np.rot90(image, i)
            new_image_extents[index,:] = extents

    return new_image_extents, new_image_data


