import cv2, csv, os, sklearn, sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Reshape
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

#_______________________________________CNN_________________________________
def nVidiaModel():
    """
    nVidea Autonomous Car Group model
    """
    model = Sequential()
    model.add(Conv2D(24,(5,5),padding='valid', activation='relu', input_shape=(80,160,3),subsample=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(36,(5,5),padding='valid', activation='relu',subsample=(2,2)))
    model.add(Conv2D(48,(5,5),padding='valid', activation='relu',subsample=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu'))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1164,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
#________________________________________PATH___DATA-SET_____CREATION____________________
def collect_DATA(dataPath, correction):
    """
    Finds all the images needed for training on the path `dataPath`.
    Combine the Centre, Left and Right images and steering angles and returns (all_Images, all_Angles)
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        #lines = getLinesFromDrivingLogs(directory)
        lines = []
        with open(dataPath + '/driving_log.csv') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                lines.append(line)

        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        images = []
        angles = []
        images.extend(center)
        images.extend(left)
        images.extend(right)
        angles.extend(measurements)
        angles.extend([x + correction for x in measurements])
        angles.extend([x - correction for x in measurements])
    return (images, angles)

#_________________________________________________________
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                #print(imagePath)
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
                #image = image[50:image.shape[0]-20,:,:]
          
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)
            
            inputs = np.array(images)
            #norm = lambda x: ((x/255.0)-0.5)
            #inputs = norm(inputs)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
#___________________________________________________________

def main():
    # Reading Data.
    images, measurements = collect_DATA(sys.argv[1], 0.2)
    print('Total Images: {}'.format( len(images)))

    # Splitting samples and creating generators.
    samples = list(zip(images, measurements))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)


    # Model creation
    model = nVidiaModel()

    # Compiling and training the model  ::   ::  
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history_object = model.fit_generator(train_generator, steps_per_epoch= round(len(train_samples)/32), validation_data = validation_generator, validation_steps=len(validation_samples), epochs=3, verbose=2)
    model.save('model.h5')
    
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])
    

if __name__ == '__main__':
    main()