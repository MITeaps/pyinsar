# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Guillaume Rongier
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from osgeo import gdal

def create_georaster_from_array(georaster_array,           
                                geotransform, 
                                projection,
                                file_type = 'MEM',
                                file_path = '',
                                data_type = gdal.GDT_Float64,
                                no_data_value = -99999.,
                                scale = 1.,
                                offset = 0.,
                                options = []):
    '''
    Create a GDAL georaster from a Numpy array
    
    @param georaster_array: The Numpy array
    @param geotransform: The extent and cell spacing of the georaster
    @param projection: The projection of the georaster
    @param file_type: Type to save the file (default is memory)
    @param file_path: Where to store the new georaster (default is memory)
    @param data_type: Data type of the georaster
    @param no_data_value: No data value for the georaster
    @param scale: Scaling factor for the georaster
    @param offset: Offset factor for the georaster
    @param options: List of options for compression
    
    @return The GDAL georaster
    '''
    georaster_x_size = georaster_array.shape[1]
    georaster_y_size = georaster_array.shape[0]
    number_of_bands = 1
    if len(georaster_array.shape) >= 3:
        georaster_x_size = georaster_array.shape[2]
        georaster_y_size = georaster_array.shape[1]
        number_of_bands = georaster_array.shape[0]

    driver = gdal.GetDriverByName(file_type)
    new_georaster = driver.Create(file_path,
                                  georaster_x_size,
                                  georaster_y_size,
                                  number_of_bands,
                                  data_type,
                                  options = options)
    new_georaster.SetGeoTransform(geotransform)
    new_georaster.SetProjection(projection)
    
    for band_number in range(1, number_of_bands + 1):
        new_georaster_band = new_georaster.GetRasterBand(band_number)
        new_georaster_band.SetNoDataValue(no_data_value)
        new_georaster_band.SetScale(scale)
        new_georaster_band.SetOffset(offset)
        # Fill the georaster band, otherwise no data values are not set in the new georaster
        new_georaster_band.Fill(no_data_value)

        if len(georaster_array.shape) >= 3:
            new_georaster_band.WriteArray(georaster_array[band_number - 1, :, :])
        else:
            new_georaster_band.WriteArray(georaster_array)

    return new_georaster