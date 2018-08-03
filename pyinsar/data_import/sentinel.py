# The MIT License (MIT)
# Copyright (c) 2018 Massachusetts Institute of Technology
#
# Author: Cody Rude, Guillaume Rongier
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

# Standard library imports
import requests
from getpass import getpass
from glob import glob
import xml.etree.ElementTree as ET

# 3rd party imports
import pandas as pd
import geopandas as gpd
from lxml import html

from pyinsar.data_import.import_utils import *

def parse_satellite_data(in_satellite_file):
    '''
    Parse Sentinel satellite data

    @param in_satellite_file: Satellite orbit filename

    @return DataFrame of orbit information
    '''
    satellite_tree = ET.parse(in_satellite_file)

    names = ['TAI', 'UTC', 'UT1','Absolute_Orbit', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Quality']
    time_converter = lambda x: pd.to_datetime(x[4:])
    converters = [time_converter, time_converter, time_converter, int, float, float, float,
                  float, float, float, lambda x: x]
    tmp_data = []

    for orbit in satellite_tree.findall('Data_Block/List_of_OSVs/OSV'):
        row = []
        for name, converter in zip(names, converters):
            row.append(converter(orbit.find(name).text))
        tmp_data.append(row)

    return pd.DataFrame(tmp_data, columns=names)

def get_url_precise_orbit(product_name):
    '''
    Get the URL of the precise orbit corresponding to a given Sentinel-1 product
    Modified from https://github.com/scottyhq/dinoSARaws/blob/48f68b0c49b26a91b501bc6d3fb1b2eb4c6c3918/bin/prep_topsApp_aws.py
    
    @param product_name: Name of the Sentinel-1 product
    
    @return The URL is the precise orbit file exists, none otherwise
    '''
    satellite = product_name[:3]
    date = product_name[17:25]
    # Incomplete inventory: https://s1qc.asf.alaska.edu/aux_poeorb/files.txt
    base_url = 'https://s1qc.asf.alaska.edu/aux_poeorb'
    r = requests.get(base_url)
    webpage = html.fromstring(r.content)
    orbits = webpage.xpath('//a/@href')
    # Get S1A or S1B
    df = gpd.pd.DataFrame(dict(orbit=orbits))
    df_sat = df[df.orbit.str.startswith(satellite)].copy()
    day_before = gpd.pd.to_datetime(date) - gpd.pd.to_timedelta(1, unit = 'd')
    day_before = day_before.strftime('%Y%m%d')
    # Get matching orbit file
    df_sat.loc[:, 'start_time'] = df_sat.orbit.str[42:50]
    match = df_sat.loc[df_sat.start_time == day_before, 'orbit'].values
    
    if len(match) == 0:
        print('No Sentinel-1 precise orbit for the product', product_name)
        return None
    return f'{base_url}/{match[0]}'
                
def download_precise_orbits(product_folder, orbit_folder, username, password):
    '''
    Download the precise orbits for all the Sentinel-1 products in a folder
    
    @param product_folder: The folder where the Sentinel-1 products are
    @param orbit_folder: The folder where to put the orbit files
    @param username: The username for authentification on Earthdata
    @param password: The password for authentification on Earthdata
    
    @return The paths of the orbit files, none if a file couldnot be downloaded
    '''
    product_paths = glob(product_folder + '/*.zip')
    precise_orbit_paths = []
    for product_path in product_paths:
        product_name = product_path.split('/')[-1].split('.')[0]
        precise_orbit_url = get_url_precise_orbit(product_name)
        if precise_orbit_url is not None:
            precise_orbit_paths.append(download_file(precise_orbit_url,
                                                     orbit_folder,
                                                     username,
                                                     password))
    
    return precise_orbit_paths

def download_products(product_names,
                      product_folder,
                      base_url = 'https://datapool.asf.alaska.edu/SLC',
                      use_vertex = True,
                      username = None,
                      password = None):
    '''
    Download Sentinel-1 products in a folder
    
    @param product_names: List of Sentinel-1 product names
    @param product_folder: The folder where to put the product files
    @param base_url: Base url from where to download the files (default is from 
                     the Alaska Satellite Facility)
    @param use_vertex: True if the base url is that of the Alaska Satellite Facility
    @param username: The username for authentification on Earthdata
    @param password: The password for authentification on Earthdata
    
    @return The paths of the orbit files, none if a file couldnot be downloaded
    '''
    if base_url[-1] != '/':
        base_url += '/'
    
    product_paths = []
    for i, product_name in enumerate(product_names):
        print('Downloading file ', i + 1, '/', len(product_names), sep = '', end = '\r')
        satellite = ''
        if product_name[0:3] == 'S1B' and use_vertex == True:
            satellite = 'SB/'
        elif product_name[0:3] == 'S1A' and use_vertex == True:
            satellite = 'SA/'
        product_url = base_url + satellite + product_name + '.zip'
        product_paths.append(download_file(product_url,
                                           product_folder,
                                           username,
                                           password))
    print('\033[K\033[K\033[K\033[K\033[K\033[K\033[KDownload over')
    
    return product_paths
