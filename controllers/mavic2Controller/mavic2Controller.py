from controller import Robot
import sys
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
from PIL import Image
import cv2
from skimage.color import rgba2rgb
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import h5py
import numpy as np
from skimage.transform import resize
from tensorflow.keras.models import load_model
from keras.layers import Conv2D, LeakyReLU


def extract_gray_roi(image, threshold=54):
    is_in_range = np.mean(image, axis=-1) < threshold
    non_color_pixels = np.where(is_in_range)

    return cv2.cvtColor(image[min(non_color_pixels[0]):max(non_color_pixels[0])
    , min(non_color_pixels[1]):max(non_color_pixels[1])], cv2.COLOR_BGR2GRAY)
    
    
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(5, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights("model_weights.weights.h5")


def get_robot_heading(compass_value):
    return math.atan2(compass_value[0], compass_value[1])
def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)
def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class Mavic (Robot):
    
    K_VERTICAL_THRUST = 68.5  
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        
    K_ROLL_P = 50.0           
    K_PITCH_P = 30.0          
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    target_precision = 0.35

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.r_led = self.getDevice("front right led")
        self.l_led = self.getDevice("front left led")
        self.camera_pitch_motor.setPosition(1.55)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)
        self.heading = 0
        self.current_pose = 6 * [0]  
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

    def set_position(self, pos):
        
        self.current_pose = pos

    def move_to_target(self, waypoints, labels, label):
        
        if self.target_position[0:2] == [0, 0]:  
            self.target_position[0:2] = waypoints[0]
            
            print("First target: ", self.target_position[0:2])
        
        
        if(calculate_euclidean_distance(self.target_position[0], self.target_position[1], self.current_pose[0]
        ,  self.current_pose[1]) < self.target_precision
         and (3.13 < abs(self.heading) < 3.15)) :
            
            
            image_data_list = []

            image = self.camera.getImage()
            
            
            if image:                    
               
                width, height = self.camera.getWidth(), self.camera.getHeight()
                
                image_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
                image = Image.fromarray(image_array)
                image.save("image.png")
                image = cv2.imread("image.png")
                
                output_image = extract_gray_roi(image)
                gray_image = np.array(output_image)
        
                gray_image = resize(gray_image, (28, 28), preserve_range=True, anti_aliasing=True)
                
                flattened_pixels = gray_image.flatten()
                gray_image = gray_image / gray_image.max()
                
                
                plt.imshow(gray_image, cmap="gray")
                plt.title("Captured Image" + str(len(flattened_pixels)))
                plt.show()
                # Add the flattened pixels along with the object name to the list
                
                image_data_list.append([labels[self.target_index]] + flattened_pixels.tolist())

                columns = ['label'] + [f'pixel_{i}' for i in range(len(image_data_list[0]) - 1)]
                df = pd.DataFrame(image_data_list, columns=columns)
                df.to_csv('test.csv', index=False)
                
                test_data = pd.read_csv('test.csv')

                X_test_real = test_data.iloc[:, 1:].values
                X_test_real = X_test_real.reshape(X_test_real.shape[0], 28, 28, 1).astype('float32') / 255.0
                
                predictions = model.predict(X_test_real)
                predicted_labels = np.argmax(predictions, axis=1)
                
                print(predicted_labels)
                
            self.target_index += 1
            self.target_position[0:2] = waypoints[self.target_index % len(waypoints)]
            print("Target reached! New target: ",
                      self.target_position[0:2])
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            if(predicted_labels[0] == label): 
            
                self.r_led.set(1);
                self.l_led.set(1);
                while waypoints[self.target_index - 1][0] - 0.4 < self.current_pose[0] < waypoints[self.target_index - 1][0] + 0.4:
                    roll, pitch, yaw = self.imu.getRollPitchYaw()
                    x_pos, y_pos, altitude = self.gps.getValues()
                    roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
                    self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
                    cx, cy, _ = self.compass.getValues()
                    self.heading = get_robot_heading((cx, cy))
                    roll_disturbance = -0.1
                    pitch_disturbance = 0
                    yaw_disturbance = 0
                    roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
                    pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
                    yaw_input = yaw_disturbance
                    clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
                    vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
    
                    front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
                    front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
                    rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
                    rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
    
                    self.front_left_motor.setVelocity(front_left_motor_input)
                    self.front_right_motor.setVelocity(-front_right_motor_input)
                    self.rear_left_motor.setVelocity(-rear_left_motor_input)
                    self.rear_right_motor.setVelocity(rear_right_motor_input)
                    self.step(self.time_step)
                self.target_altitude = 0.01
                while altitude > 0.2:
                  
                    roll, pitch, yaw = self.imu.getRollPitchYaw()
                    x_pos, y_pos, altitude = self.gps.getValues()
                    roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
                    self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
                    cx, cy, _ = self.compass.getValues()
                    self.heading = get_robot_heading((cx, cy))
                    roll_disturbance = 0
                    pitch_disturbance = 0
                    yaw_disturbance = 0
                    roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
                    pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
                    yaw_input = yaw_disturbance
                    clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
                    vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
        
                    front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
                    front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
                    rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
                    rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        
                    self.front_left_motor.setVelocity(front_left_motor_input)
                    self.front_right_motor.setVelocity(-front_right_motor_input)
                    self.rear_left_motor.setVelocity(-rear_left_motor_input)
                    self.rear_right_motor.setVelocity(rear_right_motor_input)
                    self.step(self.time_step)
                
                while self.step(self.time_step) != -1:
                    self.front_left_motor.setVelocity(0)
                    self.front_right_motor.setVelocity(0)
                    self.rear_left_motor.setVelocity(0)
                    self.rear_right_motor.setVelocity(0)
                    pass                          
        self.target_altitude = 2.8    
        

        roll_disturbance = 0.0
        
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        
        angle_left = self.target_position[2] - self.current_pose[5]
        
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
        
        
        if abs(self.target_position[0]
         - self.current_pose[0]) > 0.2 and abs(self.target_position[1]
         - self.current_pose[1]) > 0.2 : 
            if self.current_pose[0] < self.target_position[0]:
                roll_disturbance = -0.3
            else:
                roll_disturbance = 0.3
        
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        
        distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
            (self.target_position[1] - self.current_pose[1]) ** 2))
        
        return yaw_disturbance, pitch_disturbance, roll_disturbance

    def run(self):
        t1 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0
        
        
        waypoints = [[5, 0], [3, -3], [5, 0], [2, 5], [-5, 4]]
        labels =[2, 4, 3, 1, 0]
        
        self.target_altitude = 2.8
        label = 1 #target which is Tshirt
        # target not found...
        # moving towards another box
        #not found again...
        #its a shoe, soo...
        #going for another box
        # target found and we landed...
        while self.step(self.time_step) != -1:

            
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
            cx, cy, _ = self.compass.getValues()
            self.heading = get_robot_heading((cx, cy))
            if altitude > self.target_altitude - 1:
                
                if self.getTime() - t1 > 0.1:
                    yaw_disturbance, pitch_disturbance, roll_disturbance = self.move_to_target(
                        waypoints, labels, label)
                    t1 = self.getTime()

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)

robot = Mavic()
robot.run()
