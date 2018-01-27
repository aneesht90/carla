# Run game engine

./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600
or
./CarlaUE4.sh /Game/Maps/Town02 -carla-server -benchmark -fps=15 -windowed -ResX=800 -ResY=600

# Manual Driving with keyboard


# Manual Driving with Driving Controller


# Driving in autopilot model


# Collect data in autopilot mode


# Collect data in manual driving mode
python PythonClient/data_collection_controller.py -dc -vd




# Train neural network with collected data

python PythonClient/train.py -d 'Measurements/Controller/' -c 100
