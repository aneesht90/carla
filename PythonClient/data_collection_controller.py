#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB), and the INTEL Visual Computing Lab.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for carla. Please refer to client_example for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import sys
import time

try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import pandas as pd
except ImportError:
    raise RuntimeError('cannot import pandas, make sure pandas package is installed')



from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180




# make sure pygame doesn't try to open an output window
#os.environ["SDL_VIDEODRIVER"] = "dummy"

DEBUG = False
HOST = "localhost"
PORT = "50000"
WHEEL = "G27 Racing Wheel"
gear_lever_positions = {
  -1: "reverse",
  0: "neutral",
  1: "first",
  2: "second",
  3: "third",
  4: "fourth",
  5: "fifth",
  6: "sixth"
}

status_buttons = {
  #10: "parking_brake_status",
  10: "start_new_episode",
  1: "headlamp_status",
  3: "high_beam_status",
  2: "windshield_wiper_status"
}

gear_lever_position = 0
parking_brake_status = False
headlamp_status = False
high_beam_status = False
windshield_wiper_status = False
axis_mode = 1

number_of_episodes = 5
frames_per_episode = 5000







def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=15,
        NumberOfPedestrians=30,
        WeatherId=random.choice([1, 3, 7, 8, 14]))
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(200, 0, 140)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(200, 0, 140)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(200, 0, 140)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, city_name=None, drive_mode=None, store_vehicle_data=False):
        self.client = carla_client
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = city_name
        self._map = CarlaMap(city_name) if city_name is not None else None
        self._map_shape = self._map.map_image.shape if city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if city_name is not None else None
        self._drive_mode = drive_mode
        self._control = VehicleControl()
        self._control.brake = 0
        self._control.throttle = 0
        self._store_vehicle_data = store_vehicle_data
        self._datas = []
        self._episode_count = 0
    def execute(self, drive_mode=None):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        if(drive_mode):
            valid,self._wheel= self._test_drive_controller_mode(pygame)

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        scene = self.client.load_settings(make_carla_settings())
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        self._episode_count = self._episode_count + 1
        print('Starting new episode...','episode count: ',self._episode_count)
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()


        acceleration    = measurements.player_measurements.acceleration
        orientation     = measurements.player_measurements.transform.orientation
        velocity        = measurements.player_measurements.forward_speed
        if self._store_vehicle_data:
            self._datas.append([ self._timer.step,
                                 round(self._control.brake,3),
                                 round(self._control.throttle,3),
                                 round(velocity,3),
                                 round(acceleration.x,3),
                                 round(acceleration.y,3),
                                 round(acceleration.z,3),
                                 round(orientation.x,3),
                                 round(orientation.y,3),
                                 round(orientation.z,3)
                                 ])

        self._main_image = sensor_data['CameraRGB']
        self._mini_view_image1 = sensor_data['CameraDepth']
        self._mini_view_image2 = sensor_data['CameraSemSeg']

        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.get_position_on_map([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()
        if (self._drive_mode):
            #print("drive controller mode")
            #if(self._test_drive_controller_mode(pygame)):
            #    print("drive controller - works!!")
            control = self._get_drive_controller_control(pygame)
            #control = self._get_keyboard_control(pygame.key.get_pressed())
        else:
            control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.get_position_on_map([
                        measurements.player_measurements.transform.location.x,
                        measurements.player_measurements.transform.location.y,
                        measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if (control is None) :
            # storing data collected as episode data
            df = pd.DataFrame.from_records(self._datas,
                                       columns=['no',
                                                'brake',
                                                'throttle',
                                                'velocity',
                                                'acceleration-x',
                                                'acceleration-y',
                                                'acceleration-z',
                                                'orientation-x',
                                                'orientation-y',
                                                'orientation-z'
                                                ])
            save_file_name = "./Measurements/Controller/data_records_episode" + str(self._episode_count) +".csv"
            df.to_csv(save_file_name, sep=',', encoding='utf-8')
            # if self._episode_count == number_of_episodes:
            #     pygame.quit()
            self._on_new_episode()
        else:
            # if(control.throttle):
            #     control.brake = 0
            #print("test :",measurements.player_measurements.transform)
            print("brake: ",control.brake,
                  " throttle: ",control.throttle,
                  " hand brake: ",control.hand_brake,
                  " steer: ",control.steer)
            self.client.send_control(control)

    def _test_drive_controller_mode(self, pygame):
        """
        G27 steering wheel check.
        """
        #print("testing the drive controller")
        wheel = None
        for j in range(0,pygame.joystick.get_count()):
            if pygame.joystick.Joystick(j).get_name() == WHEEL:
                wheel = pygame.joystick.Joystick(j)
                wheel.init()
                #print ("Found", wheel.get_name())
                return True, wheel
        if not wheel:
            print ("No G27 steering wheel found")
            return False, None


    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        control.reverse = self._is_on_reverse
        return control

    def _get_drive_controller_control(self, pygame):
        """
        Return a VehicleControl message based on driving controller. Return None
        if a new episode was requested.
        """
        # initialising control with last time sample value
        control = self._control
        for event in pygame.event.get(pygame.JOYAXISMOTION):
            if DEBUG:
                print ("Motion on axis: ", event.axis)
            if event.axis == 0:
                control.steer = event.value
                #print("steering: ",control.steer)
            elif event.axis == 1:
                if axis_mode ==1:
                    if event.value < 0:
                        control.throttle = abs(event.value)
                        #print ("throttle: ",control.throttle)
                    else:
                        control.brake = abs(event.value)
                        #print("brake",control.brake )
                elif axis_mode == 2:
                        print("test1", self._pedal_value(event.value))
                elif axis_mode == 3:
                        print("test2", self._pedal_value(event.value))
            elif event.axis == 2 and axis_mode == 2:
                print("test3", self._pedal_value(event.value))
            elif event.axis == 3:
                print("clutch", self._pedal_value(event.value))

        for event in pygame.event.get(pygame.JOYBUTTONUP):
            if DEBUG:
                print ("Released button is", event.button)
            if (event.button >= 12 and event.button <= 17) or event.button == 22:
                gear = 0
                print("gear_lever_position", gear_lever_positions[gear])


        for event in pygame.event.get(pygame.JOYBUTTONDOWN):
            if DEBUG:
                print ("Pressed button is", event.button)
            if event.button == 0:
                print ("pressed button 0 - bye...")
                stop()
                exit(0)
            elif event.button == 11:
                print("reverse gear")
                self._is_on_reverse = not self._is_on_reverse
                control.reverse = self._is_on_reverse
                #send_data("ignition_status","start")
                #print("ignition_status - start")
            elif event.button >= 12 and event.button <= 17:
                gear = event.button - 11
                print("gear_lever_position", gear_lever_positions[gear])
            elif event.button == 22:
                gear = -1
                print("gear_lever_position", gear_lever_positions[gear])
            elif event.button in status_buttons:
                if status_buttons[event.button]== 'start_new_episode':
                    return None

        control.hand_brake  =   False
        # as same pedal is used for brake and accelerator
        if(control.brake < 0.008 or control.throttle > 0.7):
            control.brake = 0.0
        if(control.throttle < 0.008 or control.brake > 0.7):
            control.throttle = 0.0


        self._control = control
        return control

    def _pedal_value(self,value):
      '''Steering Wheel returns pedal reading as value
      between -1 (fully pressed) and 1 (not pressed)
      normalizing to value between 0 and 100%'''
      return (1 - value) * 50

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) Lane Orientation ({ori_x:.1f},{ori_y:.1f})  '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._mini_view_image1 is not None:
            array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (gap_x, mini_image_y))

        if self._mini_view_image2 is not None:
            array = image_converter.labels_to_cityscapes_palette(
                self._mini_view_image2)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self._display.blit(
                surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]
            new_window_width =(float(WINDOW_HEIGHT)/float(self._map_shape[0]))*float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos =int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos,h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.get_position_on_map([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])
                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos =int(agent_position[1] *(new_window_width/float(self._map_shape[1])))
                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos,h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-dc', '--driving-controller',
        action='store_true',
        help='use driving controller for driving')
    argparser.add_argument(
        '-vd', '--vehicle-data',
        action='store_true',
        help='save vehicle data to disk')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args.map_name, drive_mode=args.driving_controller, store_vehicle_data=args.vehicle_data)
                game.execute(drive_mode=args.driving_controller)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except Exception as exception:
            logging.exception(exception)
            sys.exit(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
