# HDLTex: [Hierarchical Deep Learning for Text Classification](https://arxiv.org/abs/1709.08267)

Refrenced paper : [HDLTex: Hierarchical Deep Learning for Text Classification](https://arxiv.org/abs/1709.08267)

DOI: [10.1109/ICMLA.2017.0-134](https://doi.org/10.1109/ICMLA.2017.0-134)

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

Linke of dataset: [DOI: 10.17632/9rw3vkcfy4.2](http://dx.doi.org/10.17632/9rw3vkcfy4.2)


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

```
$ sudo apt-get install libcupti-dev
```
**Feature Extraction:**

Global Vectors for Word Representation ([GLOVE](https://nlp.stanford.edu/projects/glove/))

        For CNN and RNN you need to download and linked the folder location to GLOVE
