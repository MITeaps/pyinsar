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

import shutil
from atomicwrites import atomic_write
import requests
import os.path

def download_file(url, folder_path, username = None, password = None, filename = None):
    '''
    Download a file from a URL
    
    @param url: The URL where the file is
    @param folder_path: Path to the folder where the downloaded file will be stored
    @param username: username for authentification, if needed
    @param password: Password for authentification, if needed
    @param filename: Change the filename, if needed
    
    @return The file path if download was succesful, none otherwise
    '''
    if filename is None:
        filename = url.split('/')[-1]
    if folder_path[-1] != '/' and filename[0] != '/':
        folder_path += '/'
    file_path = folder_path + filename
    
    if os.path.exists(file_path) == False:
        with requests.Session() as session:
            try:
                r = session.get(url, auth = (username, password), stream = True)
                r.raise_for_status()
                with atomic_write(file_path, mode = 'wb') as data_file:
                    shutil.copyfileobj(r.raw, data_file, 1024*1024*10)
                    return file_path
            except requests.exceptions.HTTPError as errh:
                print("http error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("error connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("timeout error:", errt)
            except requests.exceptions.RequestException as err:
                print("error:", err)
    else:
        print('File', filename, 'already exists in', folder_path)
        return file_path