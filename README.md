[![license](http://kowsari.net/HDLTex_DOI.svg?maxAge=2592000)](https://doi.org/10.1109/ICMLA.2017.0-134)
[![license](https://travis-ci.org/kk7nc/HDLTex.svg?branch=master)](https://travis-ci.org/kk7nc/HDLTex)
[![wercker status](https://app.wercker.com/status/24a123448ba8764b257a1df242146b8e/s/master "wercker status")](https://app.wercker.com/project/byKey/24a123448ba8764b257a1df242146b8e)
[![Join the chat at https://gitter.im/HDLTex](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/HDLTex/Lobby?source=orgpage)
[![license](https://img.shields.io/badge/arXiv-1709.08267-red.svg?style=flat)](https://arxiv.org/abs/1709.08267)
[![license](https://img.shields.io/badge/ResearchGate-HDLTex-blue.svg?style=flat)](https://www.researchgate.net/publication/319968747_HDLTex_Hierarchical_Deep_Learning_for_Text_Classification)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592104)](https://github.com/kk7nc/HDLTex/blob/master/LICENSE)





# HDLTex: Hierarchical Deep Learning for Text Classification

Refrenced paper : [HDLTex: Hierarchical Deep Learning for Text Classification](https://arxiv.org/abs/1709.08267)

![picture](http://kowsari.net/____impro/1/onewebmedia/HDLTex.png?etag=W%2F%22c90cd-59c4019b%22&sourceContentType=image%2Fpng&ignoreAspectRatio&resize=821%2B326&extract=0%2B0%2B821%2B325?raw=false "HDLTex as both Hierarchy lavel are DNN")



## Installation ##

There are git RMDL in this repository; to clone all the needed files, please use:

    git clone --recursive https://github.com/kk7nc/HDLTex.git
     
     
The primary requirements for this package are Python 3 with Tensorflow. The requirements.txt file contains a listing of the required Python packages; to install all requirements, run the following:
    
    pip -r install requirements.txt
    
Or

    pip3  install -r requirements.txt

Or:

    conda install --file requirements.txt
        
If the above command does not work, use the following:

    sudo -H pip  install -r requirements.txt

**Documentation:**

**Datasets for HDLTex:** 

Linke of dataset: [DOI: 10.17632/9rw3vkcfy4.6](http://dx.doi.org/10.17632/9rw3vkcfy4.6)


Web of Science Dataset [WOS-11967](http://dx.doi.org/10.17632/9rw3vkcfy4.2)

        This dataset contains 11,967 documents with 35 categories which include 7 parents categories.
        
Web of Science Dataset [WOS-46985](http://dx.doi.org/10.17632/9rw3vkcfy4.2)

        This dataset contains 46,985 documents with 134 categories which include 7 parents categories.
      
Web of Science Dataset [WOS-5736](http://dx.doi.org/10.17632/9rw3vkcfy4.2)

        This dataset contains 5,736 documents with 11 categories which include 3 parents categories.



**Requirment :**


General:

Python 3.5 or later see [Instruction Documents](https://www.python.org/)

TensorFlow see [Instruction Documents](https://www.tensorflow.org/install/install_linux).

scikit-learn see [Instruction Documents](http://scikit-learn.org/stable/install.html)

Keras see [Instruction Documents](https://keras.io/)

scipy see [Instruction Documents](https://www.scipy.org/install.html)

GPU:

CUDA® Toolkit 8.0. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cuda-toolkit). 

The [NVIDIA drivers associated with CUDA Toolkit 8.0](http://www.nvidia.com/Download/index.aspx).

cuDNN v6. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cudnn). 

GPU card with CUDA Compute Capability 3.0 or higher.

The libcupti-dev library,

To install this library, issue the following command:


        $ sudo apt-get install libcupti-dev


**Feature Extraction:**

Global Vectors for Word Representation ([GLOVE](https://nlp.stanford.edu/projects/glove/))

        For CNN and RNN you need to download and linked the folder location to GLOVE
        
        
## Error and Comments: ##

Send an email to [kk7nc@virginia.edu](mailto:kk7nc@virginia.edu)


## Citation: ##

    @inproceedings{Kowsari2018HDLTex, 
    author={K. Kowsari and D. E. Brown and M. Heidarysafa and K. Jafari Meimandi and M. S. Gerber and L. E. Barnes}, 
    booktitle={2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
    title={HDLTex: Hierarchical Deep Learning for Text Classification}, 
    year={2017},  
    pages={364-371}, 
    doi={10.1109/ICMLA.2017.0-134},  
    month={Dec}
    }
