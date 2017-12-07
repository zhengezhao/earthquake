# Earthquake Structural Response Visualization - Query

## Approach
User will be shown a front page which will give the topic model presentation of the simulation data
He can select a chunck on the image, which is a query
The background program will help find the top similar part in each earthquake

## Our data
quake_start.npy
topic_weights_shear.npy
the first file include the index number for each eq data in the second file

Any eq i can be get by:
################################################################
#topic_weights_shear[quake_start[i]:quake_start[i + 1], :].T
#This will give a shape(n,m), n is the topic dimension, m is the time steps dimension
################################################################

## Questions


## Getting Started
- install miniconda with python 3.6
- Add channels to search for packages:
- to create the virtual env run: conda create --file=requirements.txt -p=venv_name python=3.6
- to activate the virtual env run:source activate venv_name

## Running

- After that, you can run query.py
