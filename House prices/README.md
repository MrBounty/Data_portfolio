# Introduction
The House Prices: Advanced Regression Techniques Kaggle competition is a popular machine learning challenge in which participants are tasked with predicting the final sale price of homes in Ames, Iowa, based on a set of features describing various aspects of the properties. The competition dataset contains 79 explanatory variables, including information about the house's size, location, age, and quality, as well as other factors such as the presence of amenities like fireplaces and pools.

Participants are expected to develop models that accurately predict the sale price of each home in the test dataset based on these features, using techniques such as regression analysis and feature engineering. The competition is a great opportunity for data scientists and machine learning enthusiasts to showcase their skills and learn from other participants in the community.

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

My model for this competition used deep learning technique and neural network. I used python keras packages 

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
