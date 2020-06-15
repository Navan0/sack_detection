# sack_detection

## install dependencies

`pip install tensorflow==1.15.0`

`pip install opencv-python`


git clone https://github.com/tensorflow/models.git


/models/research/
protoc object_detection/protos/*.proto --python_out=.

## run program

`sudo apt-get install protobuf-compiler python-pil python-lxml`
`sudo pip install jupyter`
`sudo pip install matplotlib`

from sack_detection/ `protoc object_detection/protos/*.proto --python_out=.`

`sudo python setup.py build`
`sudo python setup.py install`

`export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

change paths in `main/runs.py` from `model-files`

`python runs.py`


