from numpy import argmax
from pandas import read_csv
from datetime import datetime
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

class AI(object):

    def __init__(self, parent, epoch, batch_div):
        self.parent = parent #The GUI app
        self.epoch_count = epoch #Number of epoch
        self.batch_div = batch_div #Batches divider
        self.can_predict = False

    def importData(self, train_path, test_path):
        #Import the data to and pandas table
        self.raw_train = read_csv(train_path)
        self.raw_test = read_csv(test_path)
        self.raw_train = self.raw_train.sample(frac=1) #Shuffle the data

    def start_training(self):
        self.prepare_data()
        self.make_batches()
        self.make_model()
        self.run_training()
        self.can_predict = True

    def make_batches(self):
        #Make the 2 batches to return image and value
        self.gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
        self.batches = self.gen.flow(self.x_train, self.y_train, batch_size=32)
        self.val_batches = self.gen.flow(self.x_val, self.y_val, batch_size=32)

    def make_model_old(self):
        #Make the keras model
        input_tensor = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_tensor = Dense(10, activation='softmax')(x)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)
        
    def make_model(self):
        #Make the keras model
        input_tensor = Input(shape=(28, 28, 1))
        x = Conv2D(64, (3, 3), activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_tensor = Dense(10, activation='softmax')(x)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)

    def run_training(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.batches, steps_per_epoch=len(self.x_train)//self.batch_div, validation_data=self.val_batches, validation_steps=len(self.x_val)//self.batch_div, epochs=self.epoch_count)

        #make some button of the app green and usable
        self.parent.lunch_training_button.configure(bg = "#90EE90")
        self.parent.save_model_button.config(state='normal')
        #self.parent.predict_button.config(state='normal')
        self.can_predict = True

    def prepare_data(self):
        ## Reshape and transform to numpy
        x_train = self.raw_train.to_numpy()[:, 1:]/255
        y_train = self.raw_train.to_numpy()[:, 0]
        x_train = x_train.reshape((42000, 28, 28, 1))
        x_train = x_train.astype('float32')
        y_train = y_train.astype('int32')

        ## Separate data
        self.x_val = x_train[38000:]
        self.y_val = y_train[38000:]
        self.x_train = x_train[:38000]
        self.y_train = y_train[:38000]

        #Change from [0-255] to [0-1]
        x_test = self.raw_test.to_numpy()/255
        x_test = x_test.reshape((28000, 28, 28, 1))
        self.x_test = x_test.astype('float32')

    def save_model(self):
        path = './saved_models/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.h5'
        path = path.replace(" ", "_")
        path = path.replace(":", "_")
        path = path.replace("-", "_")
        self.model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.can_predict = True

    def predict(self, image):
        result = self.model.predict(image)
        self.parent.result_text.config(text = 'Result: ' + str(argmax(result)))