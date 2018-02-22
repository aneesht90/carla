## Run game engine
./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
or
./CarlaUE4.sh /Game/Maps/Town02 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600


## Driving in autopilot model
python PythonClient/client_example.py


# Collect data in manual driving mode
## with drive controller
python PythonClient/data_collection_controller.py -dc -vd  

## keyboard control
python PythonClient/data_collection_controller.py  -vd  




## Train neural network with collected data
python PythonClient/train.py -d 'Measurements/Controller/Preprocessed' -c 6000 -g 6000

## Run the trained networking
python PythonClient/run.py -dc
