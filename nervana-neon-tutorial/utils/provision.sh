#!/bin/bash

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# activate neon virtual env
cd /home/ubuntu/neon
. .venv/bin/activate
cd /home/ubuntu/meetup

# ensure notebooks are read only
chmod u-w cifar_example.ipynb
chmod u-w imdb_example.ipynb

# launch notebook server
ipython notebook --ip 0.0.0.0 --no-browser

