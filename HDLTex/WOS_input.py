"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
HDLTex: Hierarchical Deep Learning for Text Classification
Script to download and extract Web Of Science datasets

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  HDLTex project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : HDLTex: Hierarchical Deep Learning for Text Classification
* Link: https://doi.org/10.1109/ICMLA.2017.0-134
* Comments and Error: email: kk7nc@virginia.edu
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import print_function

import os, sys, tarfile
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape


# path to the directory with the data
DATA_DIR = '.\data_WOS'

# url of the binary data
DATA_URL = 'https://www.researchgate.net/profile/Kamran-Kowsari/publication/321038556_Web_Of_Science_Dataset/data/5a09f9daaca272d40f412017/Dataset.zip?_sg%5B0%5D=T2IX7UKFm_80V4eGOmcEHFMZtHsfBS6p-MygLIgLue98TNFPiXVMFnGx5pK4e3eAinN4Z262MwNq2w-Gtzo5tg.iy1QPikF7AeR3p2iJ887KoJAQN1DvSCD1oUiDjAsA5ib8mgfdaDPXxqeWlzJ6et-PqiMabXc5QItGMERJV4VOA&_sg%5B1%5D=avnfE9AjAykTfiJF4GtikC-t-Y7pjrrh6yHA9IyEqdSAoGnIAOpEMruo8L3cEO3110HUU6XVxPNMvIJniYf5Mp5P5Dg6gQgLTlp14INYDaki.iy1QPikF7AeR3p2iJ887KoJAQN1DvSCD1oUiDjAsA5ib8mgfdaDPXxqeWlzJ6et-PqiMabXc5QItGMERJV4VOA&_iepl='


# path to the binary train file with image data


def download_and_extract():
    """
    Download and extract the WOS datasets
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)


    path = os.path.abspath(dest_directory)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)

        print('Downloaded', filename)

        tarfile.open(filepath, 'r').extractall(dest_directory)
    return path
