# Interekt

**MOVE FROM GITHUB ([https://github.com/hchauvet/RootStemExtractor](https://github.com/hchauvet/RootStemExtractor)) TO THE INRA Gitlab**

Plant motion tracker in python. Extract skeleton of simple rod-shaped object like plants to follow their motions.

This an early public release of the code, a documentation is in preparation.

![Screenshot](https://github.com/hchauvet/RootStemExtractor/raw/master/img/Screenshot1.png "screenshot")

# ChangeLog

* version 06092019:
  Passage au format hdf5 pour stocker les données, nombreuses
  amélioration et ajout de fonctions pour le phénotypage. 
  
* Version 24092018:
  Remove test dectection (Bug with Thread and matplotlib!)
  Correct bug when only one image is loaded, now the processing could be launched.
  
* Version 18052018: 
  Change multiprocessing process (Windows user can now use multiprocessing).
  Change the value of exploration diameter from 0.9 to 1.4 (line 728 of MethodOlivier in new_libgravimacro.py) 
  Remove bug with None values 

# Install

For Mac and Windows user, the simplest way to install it is to download the Anaconda distribution of python (use the branch 2.x of python)

https://www.continuum.io/downloads

## Python libraries required

Their are all part of the Anaconda python distribution

* pylab (scipy/numpy)
* pandas
* matplotlib
* scikit-image 
* h5py
* opencv (optional, but increase processing speed)

# Run

## Windows

double click on **Windows_Interekt.bat** file

## On Linux or Mac

open a terminal, go to the RootStemExtractor directory and run `python ./interekt.py`
