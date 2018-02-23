#Using python interface.

- create a virtual environment.
- source it
```bash
pip install virtualenv
mkdir carla
cd carla
virtualenv carla
pip install -r requirements.txt
```

## Run game engine in one terminal
./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
or
./CarlaUE4.sh /Game/Maps/Town02 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600


## Driving in autopilot model
python PythonClient/client_example.py


# Data Collection for training

## with drive controller G27 Racing Wheel
python PythonClient/data_collection_controller.py -dc -vd  

## keyboard control
python PythonClient/data_collection_controller.py  -vd  


# Traing and running

## Train neural network with collected data
python PythonClient/train.py -d 'Measurements/Controller/Preprocessed' -c 6000 -g 6000

## Run the trained networking
python PythonClient/run.py -dc
