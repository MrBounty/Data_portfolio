# Introduction
The House Prices: Advanced Regression Techniques Kaggle competition is a popular machine learning challenge in which participants are tasked with predicting the final sale price of homes in Ames, Iowa, based on a set of features describing various aspects of the properties. The competition dataset contains 79 explanatory variables, including information about the house's size, location, age, and quality, as well as other factors such as the presence of amenities like fireplaces and pools.

Participants are expected to develop models that accurately predict the sale price of each home in the test dataset based on these features, using techniques such as regression analysis and feature engineering. The competition is a great opportunity for data scientists and machine learning enthusiasts to showcase their skills and learn from other participants in the community.

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

For this challenge, I utilized deep learning techniques and a neural network approach using the Keras package in Python to develop a solution for predicting the sale price of houses.

# Import packages
```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from keras import Model, Input
from keras.layers import Dense, Dropout
from tqdm.keras import TqdmCallback
```

# Data import and transformation
```python
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

cat = df_train.loc[:, df_train.columns != "SalePrice"].select_dtypes(include=['object']).fillna("None")
enc = OneHotEncoder(handle_unknown='ignore').fit(cat)
cat = enc.transform(cat).toarray()
cat_sub = enc.transform(df_test.select_dtypes(include=['object']).fillna("None")).toarray()

X = df_train.loc[:, df_train.columns != "SalePrice"].select_dtypes(include=['float64', 'float', 'int']).fillna(df_train.mean()).to_numpy()
X_sub = df_test.select_dtypes(include=['float64', 'float', 'int']).fillna(df_test.mean()).to_numpy()
y = df_train["SalePrice"].to_numpy()

sc = StandardScaler()
X = sc.fit_transform(X)
X_sub = sc.transform(X_sub)

X = np.concatenate((X, cat), axis = 1)
X_sub = np.concatenate((X_sub, cat_sub), axis = 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

This code performs data preprocessing for my machine learning model to predict the SalePrice of houses.

Here's a step-by-step explanation of what's happening:

1. The first two lines of code load the training and test datasets from CSV files into pandas dataframes
2. The third line extracts all the categorical columns from the training data (i.e., columns that contain non-numeric data) and fills missing values with the string "None".
3. The fourth line creates a OneHotEncoder object and fits it to the categorical data from the training set. This encoder will be used to convert the categorical data into a one-hot encoded format that can be used as input to a machine learning model.
4. The fifth line applies the encoder to the categorical data from both the training and test sets, and converts the resulting sparse matrix to a dense array.
5. The next three lines extract the numerical features from the training and test datasets, fill any missing values with the mean of each column, and convert the resulting dataframes to numpy arrays.
6. The seventh and eighth lines apply StandardScaler to scale the numerical features, which will help the machine learning model converge more quickly during training.
7. The last two lines concatenate the scaled numerical features with the one-hot encoded categorical features, splitting the training data into training and validation sets using a test size of 0.2 and a random state of 0.
8. The resulting numpy arrays X_train, X_val, y_train, and y_val can be used as inputs to a machine learning model to predict the SalePrice of houses.

# Model
```python
number_of_input = X_train.shape[1]

input_tensor = Input(shape=(number_of_input))
x = Dense(512, activation='relu')(input_tensor)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(1)(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='mse')

history = model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          batch_size=128,epochs=100, 
          verbose=0, callbacks=[TqdmCallback(verbose=1)])
```

This code builds and trains a neural network model using the Keras API.

1. The first line calculates the number of input features in the X_train numpy array.
2. The second line creates an input tensor object for the neural network with the shape of (number_of_input).
3. The next few lines define the structure of the neural network by adding several dense layers. Each dense layer is defined with 512 units and ReLU activation, followed by a dropout layer with a dropout rate of 0.5.
4. The final layer is a dense layer with one unit, which will output the predicted SalePrice of the house.
5. The next line creates a Model object that defines the input and output tensors of the neural network.
6. The following line compiles the model by specifying the optimizer to use (Adam) and the loss function (mean squared error) to minimize during training.
7. The next line trains the neural network using the fit() method of the model object. The training data is specified as X_train and y_train, and the validation data is specified as (X_val, y_val). The batch size is set to 128, and the model is trained for 100 epochs. The TqdmCallback is used to display the training progress in a progress bar.
8. After the model is trained, the history object contains the loss values for both the training and validation sets over the course of training, which can be used to evaluate the performance of the model.

# Submission
```python
prediction = model.predict(X_sub)
prediction = prediction.reshape(1459, 1)
ids = df_test["Id"].to_numpy().reshape((1459, 1))
sub = np.concatenate((ids, prediction), axis = 1)
sub_df = pd.DataFrame(sub, columns=["Id", "SalePrice"])
sub_df["Id"] = sub_df["Id"].astype("int")
sub_df.to_csv("submission.csv", index = False)
```

This code predicts the SalePrice of houses in the test dataset using the trained neural network model and generates a submission file in CSV format.

1. The first line uses the predict() method of the trained neural network model to predict the SalePrice of houses in the test dataset, and stores the predictions in the 'prediction' numpy array.
2. The second line reshapes the 'prediction' array to have dimensions of (1459, 1).
3. The third line extracts the 'Id' column from the 'df_test' dataframe and reshapes it to have dimensions of (1459, 1).
4. The fourth line concatenates the 'ids' and 'prediction' arrays along the horizontal axis using the concatenate() function from numpy.
5. The fifth line creates a pandas dataframe 'sub_df' from the concatenated numpy array, with column names "Id" and "SalePrice".
6. The sixth line casts the 'Id' column of the 'sub_df' dataframe to an integer data type.
7. The final line writes the 'sub_df' dataframe to a CSV file named "submission.csv", with the index column excluded from the file using the 'index = False' parameter. This file can be submitted as a prediction of the SalePrice for the test dataset to a Kaggle competition or used for other purposes.
