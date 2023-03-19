# Introduction
**Note: I did not upload my code yet because the challenge is not finish**

The EY Open Science Data Challenge Level 1 aims to create open-source solutions to address UN Sustainable Development Goal 2: Zero Hunger. Participants will use radar and optical satellite data to build a model that identifies rice crops in Vietnam. Vietnam is a leading rice producer and is highly vulnerable to climate change.   


The challenge is scalable and has the potential to transform global rice production. The challenge requires math and coding skills and completing it will improve skills in Python for data science, machine learning, and managing large volumes of data. Participants will predict the presence of rice crops in a specific province of Vietnam using satellite data, with opportunities for improvement including choice of data and dataset, data preprocessing, model design and hyperparameter tuning. The output will be a rice crop classification model that can prioritize actions to ensure food security.

# My solution
I created my solution to identify rice crops in Vietnam using deep learning technologies with Keras. Using data solely from the satellite Sentinel 1, my model achieved an **accuracy of 100**%. It took me around 30 hours to get to 100%, but I also spend around 20 hours making optimization and visualization.  

In this section I will explain the workflow I used to make my model. Starting with what data I use and how I imported it. Then explain the transformation and preprocessing that I applied to it. To continue with a presentation of my Deep Learning model and how it has been train. To finish by quickly explain how I used my model to make prediction and showing a visualization of what my model is capable of.

## Import the data
Choosing the right data in a machine learning problem is arguably the most important step. It can greatly impact the accuracy and effectiveness of the resulting model. In this particular case, the challenge involved using data from multiple satellites to make predictions. The different data sources included temperature readings, vegetation indices, and others. However, after trying out various combinations, the decision was made to focus solely on RVI data (a vegetation indices). This was due to the fact that cloud interference was minimized, and the resulting curve was consistent. Additionally, the RVI data provided enough information to make accurate predictions. By carefully selecting the most appropriate data, the resulting model was able to achieve better performance and generate more meaningful insights.

Here the workflow of how I import the data that my model will use:
*	Load the csv file with 2 columns: coordinate (latitude and longitude) and rice crop presence (Yes or No)
*	Define a square box of a chosen size around each coordinate
*	Use the box to extract data from Sentinel 1 satellite for all available dates in 2022
*	Obtain an image of radio wave values and calculate the vegetation index (RVI) to represent the amount of vegetation using this formula: $\sqrt{\frac{\text{vv}}{\text{vv}+\text{vh}}} \times \frac{4 \times \text{vh}}{\text{vv}+\text{vh}}$.
*	Compute the mean value of the RVI image
*	For each coordinate, obtain an array of RVI values and an array of dates when these values were taken.

Here an image of the window use to calculate the mean RVI (the one with less resolution). And another one to give an idea of the size of it, the red box is the window use:
![Picture9](https://user-images.githubusercontent.com/79221338/226183928-c27ada4a-440d-4728-af73-8d9404799a7f.jpg)

Here an plot of the mean RVI value over time for coordinate with and without rice:   
As we can see on those graph, coordinate with rice follow a periodic curve because of the life cycle of the rice.  That is what my model will try to use to predict if there is rice at a certain coordinate.

![Picture5](https://user-images.githubusercontent.com/79221338/226183656-5be9d232-e94f-4004-9ffc-f1bf4add7da1.jpg)


Here an image to understand the life cycle of rice crops:  
<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226185780-9fd69492-ee59-4b3e-add2-42bcd78fbb1d.png">
</p>

## Transform Data
After obtaining two arrays for each coordinate (RVI and Date), we need to apply a transformation because the RVI values are unevenly spaced in time, which can lead to decreased accuracy if we train our model with this data. To address this issue, we must interpolate the RVI values using the corresponding dates, resulting in an evenly spaced array of RVI values. The interpolated curve (generated using 61 values) differs from the curves without dates or interpolation.  
After this step, we obtain an array of evenly spaced RVI values, which can be stack into a 2D array and use to train our AI. It is worth noting that we do not need to scale the data since the RVI values are already scaled during calculation.  
<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226183638-e7b0f4a3-c957-45dd-a3da-7a2781eede14.jpg">
</p>

One Issue of this challenges is the Lack of Data. Only 600 Coordinates is giving with label. To counter this, I used data augmentation. Which generate new curves slightly different from the original one, but which keeps the main characteristics, to train the model on more data and thus increase accuracy.  

When building a predictive model, it is important to ensure that the model can make accurate predictions on unseen data. To achieve this, the dataset is usually split into three separate subsets: training data, validation data, and testing data.  
*	**Training data:** This is the largest subset of the dataset and is used to train the model. The model learns from the patterns and relationships within the training data to make predictions.
*	**Validation data:** After training the model on the training data, the model's performance is evaluated on the validation data. This is used to fine-tune the model and adjust hyperparameters. The validation data is not used in training the model but is used to assess its performance and make improvements.
*	**Testing data:** The testing data is used to evaluate the final performance of the model. The testing data is completely separate from the training and validation data and is only used at the very end to evaluate the accuracy and generalizability of the model. It provides an unbiased estimate of how the model is likely to perform in the real world.  

Following the identification of a promising model, I deliberately removed the test data to augment the quantity of available training data. This approach can improve the accuracy and robustness of the model, as it provides more data for the algorithm to learn from and therefore reduces the risk of overfitting on the training set. Overall, this methodology enhances the reliability and generalizability of the model's performance.

## Create and Train Model
The model is a convolutional neural network (CNN) for processing one-dimensional (time series) data. It consists of three convolutional layers, each followed by batch normalization and a Rectified Linear Unit (ReLU) activation function. The kernel size of the first convolutional layer is 7, the second is 5, and the third is 3. The number of filters in each layer is 64, which means that each layer will output 64 feature maps.  

After the convolutional layers, a global average pooling layer is applied to compute the average of each feature map, which produces a single vector. Finally, a fully connected layer with a sigmoid activation function is used to output a single binary value, which represents the classification decision of the model.  

This kind of model is useful for tasks where the input data consists of evenly time-spaced values, such as in time series analysis, signal processing, or sensor data analysis. The CNN architecture is particularly effective for capturing temporal patterns and detecting features that may be important for making classification decisions. The use of batch normalization helps to stabilize the training process and improve the generalization performance of the model.  
```python
input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
```

Next, for the training of my model, I used this code:
```python
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    TqdmCallback(verbose=1)
]

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=0,
)
```

*	**epochs** is the number of times the model will iterate through the entire training dataset.
*	**batch_size** is the number of samples per gradient update.
*	**callbacks** is a list of callbacks that are passed to the fit() function.
*	**ModelCheckpoint** saves the best model based on the validation loss, which is useful to avoid overfitting.
*	**ReduceLROnPlateau** reduces the learning rate if the validation loss does not improve for a certain number of epochs, which helps the model to converge faster and achieve better performance.
*	**EarlyStopping** stops the training if the validation loss does not improve for a certain number of epochs, which helps to prevent overfitting and saves computational resources.
*	**TqdmCallback** is a custom callback that adds a progress bar to the training process, which is useful to monitor the progress of the training.
After defining the callbacks, the code compiles the model using the adam optimizer and binary cross-entropy loss function. It also tracks the accuracy metric during training.  

Adam is a widely used optimization algorithm in deep learning that stands for "Adaptive Moment Estimation". It's an adaptive learning rate optimization algorithm that combines the benefits of two other optimization techniques, namely, Adagrad and RMSprop. Adam has been shown to be effective in training deep neural networks, especially in cases where the data is high-dimensional and sparse.  

Binary cross-entropy is a loss function used in binary classification problems, where the goal is to predict a binary output (0 or 1). It measures the difference between the predicted output and the true output and calculates a loss value.  

The fit() function is called with the training data (x_train and y_train), validation data (x_val and y_val), and the previously defined callbacks.  
After training, the code evaluates the model on the training, validation, and testing datasets using the evaluate() function. The test loss and accuracy are printed for each dataset. My model gets 100% accuracy on all datasets.  

Here are the accuracy and loss curves of my model:  
<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226183088-2af72c16-8a4b-458e-9295-32debf5da128.png">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226183094-8a652276-1b1e-483e-8275-67f749ab5516.png">
</p>

## Submission and Visualization
After the model has been created and trained, we can leverage its predictive capabilities to detect the presence of rice by analyzing a series of 61 evenly-spaced RVI values throughout the year 2022.  

To complete this challenge, I am tasked with predicting the presence of rice crops at various coordinates listed in a CSV file. To accomplish this, I follow the same data flow by importing and preprocessing the data into an evenly time spaced RVI array before feeding it into the trained model for predictions. Finally, I generate a CSV file in the required format and upload it to the challenge's webpage to submit my results. **I get an accuracy of 100% on the 250 coordinates to predict**.  

The advantage of such a simple model is the possibility of creating predictions for whole regions and not just for coordinates. For example, I imported an RVI image and implemented an algorithm to calculate the mean RVI value within a defined area surrounding each pixel. Using this data, the model can predict whether a particular pixel contains a rice crop or not, and this prediction is then used to generate a new image, referred to as the “Mask”. In this image, pixels are colored green if a rice crop is detected, red if it is not detected, and yellow if the model is uncertain.

<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226183748-afe263fb-f68b-4996-b743-e55372003f1f.jpg">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/79221338/226183756-6e2e9d5d-3780-446d-8b5b-08345c67b8a5.jpg">
</p>

# Sources
[1] List of vegetation indices - [Implementing Sentinel-2 Data and Machine Learning to Detect Plant Stress in Olive Groves](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjd66P4mKv9AhUNH-wKHTmMBUYQFnoECBQQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F14%2F23%2F5947%2Fpdf&usg=AOvVaw2Y97GAr9QX-9fK-iIhr-8L)  
[2] Planetary computer dataset and notebook example - [Sentinel 1](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) [Landsat 2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)  
[3] Understand how to import satellite data - [Reading Data from the STAC API](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)  
[4] How to make a binary classification from timeseries using Keras - [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)  
[5] Challenges overview, data, and examples - [Level 1 – Crop identification ](https://challenge.ey.com/challenges/level-1-crop-identification-ey)
