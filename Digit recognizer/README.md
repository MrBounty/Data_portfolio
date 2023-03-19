# Introduction
The Kaggle Digit Recognizer competition is a challenging problem in the field of computer vision and machine learning. The objective of this competition is to develop a model that can accurately recognize handwritten digits from the MNIST dataset, which contains tens of thousands of images of handwritten digits.

The ability to recognize handwritten digits has a wide range of applications, from reading postal codes to recognizing the contents of financial documents. In recent years, deep learning techniques have made significant advancements in the field of image recognition, and the Kaggle Digit Recognizer competition provides an excellent opportunity to explore these techniques in practice.

In this portfolio, I will present my solution to the Kaggle Digit Recognizer competition that obtain around **99% accuracy**. My approach involves developing a convolutional neural network (CNN) architecture that is trained on the MNIST dataset using techniques such as data augmentation, regularization, and hyperparameter tuning.

Through my solution, I aim to demonstrate my skills in machine learning, deep learning, and computer vision, and how I approach and solve real-world problems.

# Application
I used the tkinter python package to make a really simple application, here an image of it:

![image](https://user-images.githubusercontent.com/79221338/226187194-39d0c9c9-b37d-4410-b81e-7c9d83a1c059.png)

Here a step by step to use the application:
1. Import the train and test file from the kaggle competition page using `Open the train file` and `Open the test file` buttons.
2. Chose the number of epoch and the batch size.
3. Train the model by cliking on the `Start training AI` button.
4. Optionally, give it a name and save it into a file by clicking on the `Save model` button.

OR  

Import an previously trained model by clicking on the `Load model` button.

Once that done, we can paint on the white zone (canvas).  
The application will give the number that is most likely on the canvas on the bottom of the application like that: `Result: 4`.  
To reset the canvas, use the `Clean` button.

Here a gif to give an idea of how it's work:

![most likely synonym - Google Search](https://user-images.githubusercontent.com/79221338/226188313-34b72d79-9bfe-4943-8a4c-87971bda58f8.gif)

# Data
Data come from the kaggle competion Digit Recognizer: https://www.kaggle.com/competitions/digit-recognizer/data

Here the code used to import and prepare the data:
```python
# Import
raw_train = read_csv(train_path)
raw_test = read_csv(test_path)
raw_train = raw_train.sample(frac=1)

# To numpy
x_train = raw_train.to_numpy()[:, 1:]/255
y_train = raw_train.to_numpy()[:, 0]
x_train = x_train.reshape((42000, 28, 28, 1))
x_train = x_train.astype('float32')
y_train = y_train.astype('int32')

# Separate data
x_val = x_train[38000:]
y_val = y_train[38000:]
x_train = x_train[:38000]
y_train = y_train[:38000]

# Change from [0-255] to [0-1]
x_test = raw_test.to_numpy()/255
x_test = x_test.reshape((28000, 28, 28, 1))
x_test = x_test.astype('float32')

# Data augmentation
self.gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
self.batches = self.gen.flow(self.x_train, self.y_train, batch_size=32)
self.val_batches = self.gen.flow(self.x_val, self.y_val, batch_size=32)
```

The dataset is read from CSV files using the read_csv function from the Pandas library.

The training data is shuffled using the sample method from Pandas to ensure that the data is randomly ordered. Then, the data is split into training and validation sets.

The pixel values of the images in the training and testing sets are scaled to a range of [0, 1] by dividing each pixel value by 255. This is done to normalize the data and make it easier for the model to learn from it.

The training and validation data are stored as NumPy arrays in the x_train, y_train, x_val, and y_val attributes of the class instance.

The testing data is stored in the x_test attribute of the class instance. It is also scaled to a range of [0, 1] using the same method as the training and validation data.

At the end, this code sets up an instance of the Keras ImageDataGenerator class to perform data augmentation on the input images.

ImageDataGenerator generates batches of image data with real-time data augmentation. The data augmentation parameters such as rotation, width shift, height shift, and zoom are used to generate new transformed images from the existing images to create additional training data. This helps to improve the generalization ability of the model and prevent overfitting.

In this specific code, the ImageDataGenerator object is initialized with several data augmentation parameters: rotation_range=8 indicates that images will be randomly rotated up to 8 degrees, width_shift_range=0.08 and height_shift_range=0.08 indicate that images will be randomly shifted horizontally and vertically up to 8% of the image size, shear_range=0.3 indicates that images will be randomly sheared up to 0.3 radians, and zoom_range=0.08 indicates that images will be randomly zoomed up to 8%.

The batches and val_batches variables are created by calling the flow method of the ImageDataGenerator class. These variables generate batches of augmented images from the training and validation sets (x_train, y_train, x_val, y_val) with a batch size of 32. These batches are then used for training and validation of the neural network model.

# Model
Here the model used by the application:
```python
input_tensor = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

The model has the following layers:
1. An input layer (Input) that takes an input tensor of shape (28, 28, 1), where 1 is the number of channels (grayscale).
2. A convolutional layer (Conv2D) with 32 filters of size (3, 3). The activation function used is the rectified linear unit (relu).
3. Another convolutional layer (Conv2D) with 64 filters of size (3, 3). The activation function used is also relu.
4. A max pooling layer (MaxPooling2D) with a pool size of (2, 2).
5. A dropout layer (Dropout) with a rate of 0.25. Dropout is a regularization technique that randomly drops out a fraction of the units in the layer during training to prevent overfitting.
6. A flattening layer (Flatten) that converts the output of the previous layer into a 1D array.
7. A dense layer (Dense) with 128 neurons and activation function relu.
8. Another dropout layer (Dropout) with a rate of 0.5.
9. An output layer (Dense) with 10 neurons (one for each digit) and activation function softmax. Softmax is used to normalize the output of the layer so that it represents a probability distribution over the 10 possible classes.

The model is compiled using the compile method with the following arguments:
* Optimizer: Adam. Adam is a popular optimizer that adapts the learning rate of the model during training.
* Loss function: Sparse categorical cross-entropy. This is the appropriate loss function to use for multi-class classification problems with integer labels.
* Evaluation metric: Accuracy. This is the metric used to evaluate the performance of the model during training and testing.
