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

import os

def create_product_xml(xml_path,
                       product_path,
                       product_type = 'master',
                       product_output_path = None,
                       product_orbit_path = None,
                       product_auxiliary_data_path = None,
                       do_add = True):
    '''
    Create the xml file defining a Sentinel-1 product for processing with ISCE
    
    @param xml_path: Path to the xml file
    @param product_path: Path to the Sentinel-1 product
    @param product_type: Master or slave product
    @param product_output_path: Path for the processing outputs of this product
    @param product_orbit_path: Path to the folder containing orbit files
    @param product_auxiliary_data_path: Path to the folder containing auxiliary data
    @param do_add: True if the component is added to an already existing xml file, false otherwise
    '''
    mode = 'w'
    prefix = ''
    if do_add == True:
        mode = 'a'
        prefix = '    '
    with open(xml_path, mode) as xml_file:
        xml_file.write(prefix + '''<component name="''' + product_type + '''">\n''')
        xml_file.write(prefix + '''  <property name="safe">["''' + product_path + '''"]</property>\n''')
        if product_auxiliary_data_path is not None:
            xml_file.write(prefix + '''  <property name="auxiliary data directory">''' + product_auxiliary_data_path + '''</property>\n''')
        if product_orbit_path is not None:
            xml_file.write(prefix + '''  <property name="orbit directory">''' + product_orbit_path + '''</property>\n''')
        if product_output_path is not None:
            xml_file.write(prefix + '''  <property name="output directory">''' + product_output_path + '''</property>\n''')
        xml_file.write(prefix + '''</component>\n''')

def create_topsApp_xml(xml_folder_path,
                       master_path,
                       slave_path,
                       master_output_path = None,
                       slave_output_path = None,
                       master_orbit_path = None,
                       slave_orbit_path = None,
                       master_auxiliary_data_path = None,
                       slave_auxiliary_data_path = None,
                       do_unwrap = True,
                       unwrapper_name = 'snaphu_mcf',
                       xml_filename = 'topsApp.xml'):
    '''
    Create the topsApp.xml file for processing Sentinel-1 data with ISCE
    
    @param xml_folder_path: Path to the folder that will contain the xml file
    @param master_path: Path to the master Sentinel-1 product
    @param slave_path: Path to the slave Sentinel-1 product
    @param master_output_path: Path for the processing outputs of the master product
    @param slave_output_path: Path for the processing outputs of the slave product
    @param master_orbit_path: Path to the folder containing orbit files for the master product
    @param slave_orbit_path: Path to the folder containing orbit files for the slave product
    @param master_auxiliary_data_path: Path to the folder containing auxiliary data for the master product
    @param slave_auxiliary_data_path: Path to the folder containing auxiliary data for the slave product
    @param do_unwrap: True to unwrap the created interferogram, false otherwise
    @param unwrapper_name: Name of the unwrapper when do_unwrap is true
    @param xml_filename: Name of the topsApp.xml file to create
    
    @return The path to the created topsApp.xml file
    '''
    xml_path = xml_folder_path
    if xml_folder_path[-1] != '/':
        xml_path += '/'
    xml_path += xml_filename
    with open(xml_path, 'w') as xml_file:
        xml_file.write('''<?xml version="1.0" encoding="UTF-8"?>\n''')
        xml_file.write('''<topsApp>\n''')
        xml_file.write('''  <component name="topsinsar">\n''')
        xml_file.write('''    <property name="Sensor name">SENTINEL1</property>\n''')
        if do_unwrap == True:
            xml_file.write('''    <property name="do unwrap">True</property>\n''')
            xml_file.write('''    <property name="unwrapper name">''' + unwrapper_name + '''</property>\n''')
    create_product_xml(xml_path,
                       master_path,
                       product_type = 'master',
                       product_output_path = master_output_path,
                       product_orbit_path = master_orbit_path,
                       product_auxiliary_data_path = master_auxiliary_data_path)
    create_product_xml(xml_path,
                       slave_path,
                       product_type = 'slave',
                       product_output_path = slave_output_path,
                       product_orbit_path = slave_orbit_path,
                       product_auxiliary_data_path = slave_auxiliary_data_path)
    with open(xml_path, 'a') as xml_file:
        xml_file.write('''  </component>\n''')
        xml_file.write('''</topsApp>\n''')
    
    return xml_path

def prepare_topsApps(product_paths,
                     result_folder_path,
                     orbit_path = None,
                     auxiliary_data_path = None,
                     do_unwrap = True,
                     unwrapper_name = 'snaphu_mcf'):
    '''
    Create a Pair_X folder for each successive pair X of Sentinel-1 product in product_paths,
    and create a topsApp.xml to process that pair with ISCE
    
    @param product_paths: List of paths to the Sentinel-1 products
    @param result_folder_path: Directory where the Pair_X folders will be created
    @param orbit_path: Path to the folder containing orbit files
    @param auxiliary_data_path: Path to the folder containing auxiliary data
    @param do_unwrap: True to unwrap the created interferogram, false otherwise
    @param unwrapper_name: Name of the unwrapper when do_unwrap is true
    
    @return The path to the created topsApp.xml file
    '''
    topsApp_paths = []
    for i in range(len(product_paths) - 1):
        result_directory = result_folder_path + 'Pair_' + str(i + 1)
        os.makedirs(result_directory, exist_ok = True)
        topsApp_paths.append(create_topsApp_xml(result_directory,
                                                product_paths[i],
                                                product_paths[i + 1],
                                                master_output_path = 'master',
                                                slave_output_path = 'slave',
                                                master_orbit_path = orbit_path,
                                                slave_orbit_path = orbit_path,
                                                master_auxiliary_data_path = auxiliary_data_path,
                                                slave_auxiliary_data_path = auxiliary_data_path,
                                                do_unwrap = do_unwrap,
                                                unwrapper_name = unwrapper_name))
        
    return topsApp_paths