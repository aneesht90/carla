#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

###
"""
From lucosanta:

This is a very rudimental way to collect data from CARLA see <https://github.com/carla-simulator/carla>

It is the carla_example.py which use ai_control data to drive around the town. Using such methodology,
the car will drive correctly and at the end, you will find data saved inside the folders. It is not the
best thing you could ever see but it works. :)

Enjoy it!

"""



from __future__ import print_function
# General Imports
import numpy as np
from PIL import Image
import random
import time
import sys
import argparse
import logging
import os
import keyboard
# Carla imports, Everything can be imported directly from "carla" package

from carla import CARLA
from carla import Control
from carla import Measurements
import pandas as pd
# Constant that set how offten the episodes are reseted

RESET_FREQUENCY = 100

from aenum import Enum


class WeatherCondition(Enum):
	Default =0
	ClearNoon = 1
	CloudyNoon= 2
	WetNoon=3
	WetCloudyNoon = 4
	MidRainyNoon = 5
	HardRainNoon = 6
	SoftRainNoon = 7
	ClearSunset = 8
	CloudySunset = 9
	WetSunset = 10
	WetCloudySunset = 11
	MidRainSunset = 12
	HardRainSunset = 13
	SoftRainSunset = 14

"""
Print function, prints all the measurements saving
the images into a folder. WARNING just prints the first BGRA image
Args:
    param1: The measurements dictionary to be printed
    param2: The iterations

Returns:
    None
Raises:
    None
"""



def print_pack(measurements,i,write_images):

	if write_images:
		image_result = Image.fromarray( measurements['BGRA'][0])

		b, g, r,a = image_result.split()
		image_result = Image.merge("RGBA", (r, g, b,a))

		if not os.path.exists('images'):
			os.mkdir('images')
		image_result.save('images/image' + str(i) + '.png')


	print ('Pack ',i)
	print ('	Wall Time: ',measurements['WallTime'])
	print ('	Game Time: ',measurements['GameTime'])
	print ('	Player Measurements ')

	print ('		Position: (%f,%f,%f)' % (measurements['PlayerMeasurements'].\
		transform.location.x,measurements['PlayerMeasurements'].transform.location.y,\
		measurements['PlayerMeasurements'].transform.location.z ))
	print ('		Orientation: (%f,%f,%f)' % (measurements['PlayerMeasurements'].\
		transform.orientation.x,measurements['PlayerMeasurements'].transform.orientation.y,\
		measurements['PlayerMeasurements'].transform.orientation.z ))

	print ('		Acceleration: (%f,%f,%f)' % (measurements['PlayerMeasurements'].\
		acceleration.x,measurements['PlayerMeasurements'].acceleration.y,measurements['PlayerMeasurements'].acceleration.z ))
	print ('		Speed: ',measurements['PlayerMeasurements'].forward_speed)
	print ('		Collision Vehicles (Acum. Impact): ',measurements['PlayerMeasurements'].collision_vehicles)
	print ('		Collision Pedestrians (Acum. Impact): ',measurements['PlayerMeasurements'].collision_pedestrians)
	print ('		Collision Other (Acum. Impact): ',measurements['PlayerMeasurements'].collision_other)
	print ('		Intersection Opposite Lane (% Volume): ',measurements['PlayerMeasurements'].intersection_otherlane)
	print ('		Intersection Opposite Sidewalk (% Volume): ',measurements['PlayerMeasurements'].intersection_offroad)


	print ('	',len(measurements['Agents']),' Agents (Positions not printed)')
	print ('		',end='')
	for agent in measurements['Agents']:

		if agent.HasField('vehicle'):
			print ('Car',end='')

		elif agent.HasField('pedestrian'):
			print ('Pedestrian',end='')

		elif agent.HasField('traffic_light'):
			print ('Traffic Light',end='')


		elif agent.HasField('speed_limit_sign'):
			print ('Speed Limit Sign',end='')
		print(',',end='')
	print('')



def use_example(ini_file,port = 2000, host ='127.0.0.1',print_measurements =False,images_to_disk=False, collect_data=False):

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the CARLA
    # constructor, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong.

	carla =CARLA(host,port)

	""" As a first step, Carla must have a configuration file loaded. This will load a map in the server
		with the properties specified by the ini file. It returns all the posible starting positions on the map
		in a vector.
	"""
	positions = carla.loadConfigurationFile(ini_file)

	"""
		Ask Server for a new episode starting on position of index zero in the positions vector
	"""
	carla.newEpisode(0)

	capture = time.time()
	# General iteratior
	i = 1
	# Iterator that will go over the positions on the map after each episode
	iterator_start_positions = 1
	tmp_throttle = 0


	if not os.path.exists('images'):
		os.mkdir('images')
	datas= []
	images = []

	weather_condition_change = WeatherCondition.ClearNoon
	ending = False
	print(weather_condition_change.name.lower())

	while True:
		try:
			"""
				User get the measurements dictionary from the server.
				Measurements contains:
				* WallTime: Current time on Wall from server machine.
				* GameTime: Current time on Game. Restarts at every episode
				* PlayerMeasurements : All information and events that happens to player
				* Agents : All non-player agents present on the map information such as cars positions, traffic light states
				* BRGA : BGRA optical images
				* Depth : Depth Images
				* Labels : Images containing the semantic segmentation. NOTE: the semantic segmentation must be
					previously activated on the server. See documentation for that.

			"""
			measurements = carla.getMeasurements()



			# Print all the measurements... Will write images to disk
			#if print_measurements:
			#print_pack(measurements,i,False)

			"""
				Sends a control command to the server
				This control structue contains the following fields:
				* throttle : goes from 0 to -1
				* steer : goes from -1 to 1
				* brake : goes from 0 to 1
				* hand_brake : Activate or desactivate the Hand Brake.
				* reverse: Activate or desactive the reverse gear.

			"""

			if collect_data == True:
				print('Collecting data')
				ai = measurements['PlayerMeasurements'].ai_control
				control = Control()
				if measurements['PlayerMeasurements'].forward_speed < 50:
					control.brake = ai.brake
				else:
					control.brake = 1.0
				control.throttle = ai.throttle
				control.steer = ai.steer

				control.hand_brake = ai.hand_brake
				control.reverse = ai.reverse

				carla.sendCommand(control)
				print('Sent data',i)




				if (i % 10) ==0:
					print('Collecting data',i)
					images.append(measurements['BGRA'][0])



					# print(measurements)


					if ai.brake is None:
						control.brake = 0.0
					if ai.steer is None:
						control.steer = 0.0
					if ai.throttle is None:
						control.throttle = 0.0
					if ai.hand_brake is None:
						control.hand_brake = 0
					else:
						control.hand_brake = 1
					if ai.reverse is None:
						control.reverse = 0
					else:
						control.reverse = 1

					datas.append([ i,control.brake, control.steer,control.throttle,control.hand_brake,control.reverse])


					if random.randint(1,500) > 450:
						print('Press c')
						keyboard.press('c')



			else:
				control = Control()
				control.throttle = 0.9
				control.steer = 0

				carla.sendCommand(control)



			i+=1


			if i == 2000000:



				print('Fps for this episode : ', (1.0 / ((time.time() - capture) / 100.0)))

				"""
                    Starts another new episode, the episode will have the same configuration as the previous
                    one. In order to change configuration, the loadConfigurationFile could be called at any
                    time.
                """


				for k in range(0, len(images)):
					image_result = Image.fromarray(images[k])
					b, g, r, a = image_result.split()
					image_result = Image.merge("RGBA", (r, g, b, a))

					if not os.path.exists('images'):
						os.mkdir('images')
					image_result.save('images/image' + str(k) + '.png')

				df = pd.DataFrame.from_records(datas,
											   columns=['no', 'brake', 'steer', 'throttle', 'hand_brake', 'reverse'])
				df.to_csv('./data_records.csv', sep=',', encoding='utf-8')

				if ending :
					quit()


			'''
			if i % RESET_FREQUENCY ==0:

				print ('Fps for this episode : ',(1.0/((time.time() -capture)/100.0)))


				"""
					Starts another new episode, the episode will have the same configuration as the previous
					one. In order to change configuration, the loadConfigurationFile could be called at any
					time.
				"""
				if iterator_start_positions < len(positions):
					carla.newEpisode(iterator_start_positions)
					iterator_start_positions+=1
				else :
					carla.newEpisode(0)
					iterator_start_positions = 1

				print("Now Starting on Position: ",iterator_start_positions-1)
				capture = time.time()

			'''


		except Exception as e:

			logging.exception('Exception raised to the top')
			time.sleep(1)



if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Run the carla client example that connects to a server')
	parser.add_argument('host', metavar='HOST', type=str, help='host to connect to')
	parser.add_argument('port', metavar='PORT', type=int, help='port to connect to')

	parser.add_argument("-c", "--config", help="the path for the server ini file that the client sends",type=str,default="CarlaSettings.ini")


	parser.add_argument("-l", "--log", help="activate the log file",action="store_true")
	parser.add_argument("-lv", "--log_verbose", help="activate log and put the log file to screen",action="store_true")

	parser.add_argument("-pm", "--print", help=" prints the game measurements",action="store_true")
	parser.add_argument("--collect-data", help=" prints the game measurements", action="store_true")
	parser.add_argument(
		'-i', '--images-to-disk',
		action='store_true',
		help='save images to disk')


	args = parser.parse_args()
	if args.log or args.log_verbose:
		LOG_FILENAME = 'log_manual_control.log'
		logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

		if args.log_verbose:  # set of functions to put the logging to screen


			root = logging.getLogger()
			root.setLevel(logging.DEBUG)
			ch = logging.StreamHandler(sys.stdout)
			ch.setLevel(logging.DEBUG)
			formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
			ch.setFormatter(formatter)
			root.addHandler(ch)
	else:
		sys.tracebacklimit=0 # Avoid huge exception messages out of debug mode



	use_example(args.config,
			port=args.port,
	 		host=args.host,
	 		print_measurements=args.print,
	 		images_to_disk= args.images_to_disk,
	 		collect_data=True)
