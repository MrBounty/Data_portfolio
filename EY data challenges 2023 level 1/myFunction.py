import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

def proba_to_bool(pred):
    '''
    Transform the probability return by the model into a boolean (0.5 to 1 into 1, 0 to 0.5 into 0)
    
    Args:
        pred (ndarray): A 1D array of float
        
    Return:
        pred (ndarray): A 1D array of boolean
    '''
    for i in range(len(pred)):
        pred[i] = bool(round(pred[i][0]))
    return pred

def bool_to_string(y):
    '''
    Transform boolean into string for submission (1 for Rice, 0 for Non Rice)
    
    Args:
        y (ndarray): A 1D array of boolean
        
    Return:
        y (ndarray): A 1D array of string
    '''
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = "Rice"
        elif y[i] == 0:
            y[i] = "Non Rice"
        else:
            print("error at {i}")
    return y

def dateTime_to_second(arr):
    '''
    Transform a array of date time into an array of second
    
    Args:
        arr  (ndarray): A 2D array of date time, axis 0 is sample and axis 1 is date
        
    Return:
        arr2 (ndarray): A 2D array of second, axis 0 is sample and axis 1 is date
    '''
    epoch = arr.min()
    div = np.timedelta64(1, "s")
    arr2 = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        arr2[i, :] = (arr[i, :] - epoch) / div
    return arr2

def interpol_X(X, t, step=500000):
    '''
    Performs linear interpolation of the input array X at evenly spaced time points defined by the step argument and returns the interpolated values as a new numpy array
    
    Args:
        X (ndarray):  A 2D array of RVI value, axis 0 is sample and axis 1 is RVI value
        t (ndarray):  A 2D array of date time when RVI value have been take, axis 0 is sample and axis 1 is date
        step (float): Number of second between RVI value, 500000 is around 6 days
        
    Return:
        arr2 (ndarray): A 2D array of RVI value at evenly spaced time points, axis 0 is sample and axis 1 is RVI value
    '''
    # Transform the date time into second
    t_s = dateTime_to_second(t)
    
    # Create an evenly distribute array and find how many value will be return by the interpolation
    arr = np.arange(t_s.min(), t_s.max(), step)
    nb_value = np.interp(arr, t_s[0, :], X[0, :]).shape[0]
    
    # Find interpolation and put it into an array
    arr2 = np.zeros((X.shape[0], nb_value))
    for i in range(X.shape[0]):
        arr2[i, :] = np.interp(arr, t_s[i, :], X[i, :])
    return arr2

def make_model(input_shape):
    '''
    Return a keras model use to train and predict the presence of rice or not
    
    Args:
        input_shape (int): The number of value of RVI use as input
        
    Return:
        model (Functional): The keras model
    '''
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same", activation = "relu")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation = "relu")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation = "relu")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def submission_from_model(model, X):
    '''
    Return and save into a csv the submision file 
    
    Args:
        model (Functional): The trained keras model 
        X (ndarray):        A 2D array with the RVI value of each coordinate to evaluate
        
    Return:
        submission_df (DataFrame): The DataFrame with 2 column, id: a string with the coordinates of the point evaluate and target: A string "Rice" or "Non Rice"
    '''
    # Import the file with coordinate
    test_file = pd.read_csv('data/challenge_1_submission_template.csv')
    
    # Making predictions
    y = proba_to_bool(model.predict(X))
    y = [item for sublist in y for item in sublist]
    y = bool_to_string(y)
    y = np.array(y)
    
    # Combining the coordinate and predictions into dataframe
    submission_df = pd.DataFrame({'id':test_file['Latitude and Longitude'].values, 'target':y})
    
    # Save the prediction into a file for submission
    submission_df.to_csv("challenge_1_submission_rice_crop_prediction.csv",index = False)
    
    return submission_df

def plot_loss(history):
    '''
    Plot the loss curve of the trained model
    
    Args:
        history (history): History of the trained keras model
    '''
    df = pd.DataFrame(history.history)

    plt.figure(figsize=(15,5))
    sns.lineplot(data=df["loss"],lw=3)
    sns.lineplot(data=df["val_loss"],lw=3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    sns.despine()
    
def plot_accuracy(history):
    '''
    Plot the accuracy curve of the trained model
    
    Args:
        history (history): History of the trained keras model
    '''
    df = pd.DataFrame(history.history)

    plt.figure(figsize=(15,5))
    sns.lineplot(data=df["accuracy"],lw=3)
    sns.lineplot(data=df["val_accuracy"],lw=3)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    sns.despine()
    
#----------------------------------------------------------------------------------------------

def get_box(lat, lon, box_size):
    """
    Returns a tuple of four coordinates that represent the edges of a box centered around a given latitude and longitude.
    
    Args:
        lat (float):      The latitude of the center of the box.
        lon (float):      The longitude of the center of the box.
        box_size (float): The size of the box in degrees.

    Returns:
        Tuple[float, float, float, float]: A tuple of four floats representing the minimum and maximum values of latitude and longitude that define the edges of the box.
    """
    min_lon = lon-box_size/2
    min_lat = lat-box_size/2
    max_lon = lon+box_size/2
    max_lat = lat+box_size/2
    return (min_lon, min_lat, max_lon, max_lat)

def latlong_to_rvi(df, catalog, date, box_size_deg, resolution, cond=""):
    """
    Return a list of numpy array with the mean of RVIS and the date it been taken
    
    Args:
        df (DataFrame):       DataFrame with a column "Latitude and Longitude"
        catalog (Client):     Planetary computer catalog
        date (string):        The time range between which the data is to be taken. It is in the format of "start_date/end_date" (e.g., "2021-01-01/2022-12-31")
        box_size_deg (float): The size of the bounding box to extract data from
        resolution (float):   The resolution of the satelite data
        cond (string):        A string used for printing purposes to indicate the type of data returned (e.g., "Rice", "Non-rice", "Submission")
        
    Returns:
        rvis  (list):  List of mean RVI value
        times (list) : List of date and time of when the mean RVI value have been taken
    """
    
    # Get a list of all latitude and longitude values from the DataFrame
    lat_long = []
    for cc in df["Latitude and Longitude"]:
        latlong = cc.replace('(','').replace(')','').replace(' ','').split(',')
        lat_long.append((float(latlong[0]), float(latlong[1])))
        
    # Convert the list of latitude and longitude values into a list of bounding boxes
    bboxs = []
    for cc in lat_long:
        bboxs.append(get_box(cc[0], cc[1], box_size_deg))
        
    # Query the catalog for all the satellite items within the specified time range and the corresponding bounding boxes
    items = []
    print("Import items from catalog for " + cond)
    for bbox in tqdm(bboxs):
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=date)
        items.append(list(search.get_all_items()))
        
    # Convert the queried items into the desired data format, which includes the "vv" and "vh" bands in a 2D image
    datas = []
    print("Convert item to data for " + cond)
    for item, bbox in tqdm(zip(items, bboxs), total=len(items)):
        datas.append(stac_load(item,bands=["vv", "vh"], patch_url=pc.sign, bbox=bbox, crs="EPSG:4326", resolution=resolution))
        
    # Calculate the mean of the "vv" and "vh" bands for each image, turning a list of 2D images into a list of floats
    means = []
    print("Compute data to mean for " + cond)
    for data in tqdm(datas):
        means.append(data.mean(dim=['latitude','longitude']).compute())
        
    # Calculate the mean RVI value using the mean of "vv" and "vh" bands, and return it in a list along with the date when it was taken
    rvis = []
    times = []
    print("Calcul rvi for " + cond)
    for mean in tqdm(means):
        times.append(mean.time.to_numpy())
        rvis.append(((np.sqrt((mean.vv / (mean.vv + mean.vh))))*((4*mean.vh)/(mean.vv + mean.vh))).to_numpy())
        
    return rvis, times

def import_and_save_X_sub(date, box_size_deg, resolution, catalog):
    '''
    Import RVI and date from the satelite sentinel 1 of submission value, turn them into numpy array and save them in a file
    
    Args:
        catalog (Client):     Planetary computer catalog
        date (string):        The time range between which the data is to be taken. It is in the format of "start_date/end_date" (e.g., "2021-01-01/2022-12-31")
        box_size_deg (float): The size of the bounding box to extract data from
        resolution (float):   The resolution of the satelite data
    '''
    # Open the file with latitude and longitude into a pandas DataFrame
    test_file = pd.read_csv('data/challenge_1_submission_template.csv')
    
    # Get Sentinel-1-RTC Data
    rvi, times = latlong_to_rvi(test_file, catalog, date, box_size_deg, resolution, "Submission data")
    
    # Change list into a numpy array
    rvi = np.array(rvi)
    times = np.array(times)
    
    # Save the numpy array into a file
    np.save("rvi_sub.npy", rvi)
    np.save("times_sub.npy", times)

def import_and_save_train_data(date, box_size_deg, resolution, catalog):
    '''
    Import RVI and date from the satelite sentinel 1 of train value, turn them into numpy array and save them in a file
    
    Args:
        catalog (Client):     Planetary computer catalog
        date (string):        The time range between which the data is to be taken. It is in the format of "start_date/end_date" (e.g., "2021-01-01/2022-12-31")
        box_size_deg (float): The size of the bounding box to extract data from
        resolution (float):   The resolution of the satelite data
    '''
    # Open the file with latitude and longitude into a pandas DataFrame
    df = pd.read_csv("data/Crop_Location_Data.csv")

    # Get Sentinel-1-RTC Data
    # For coordinate with rice
    rvis_rice, times_rice = latlong_to_rvi(df.loc[df["Class of Land"] == "Rice"], catalog, date, box_size_deg, resolution, "Rice")
    # For coordinate without rice
    rvis_nrice, times_nrice = latlong_to_rvi(df.loc[df["Class of Land"] == "Non Rice"], catalog, date, box_size_deg, resolution, "Non Rice")
    
    # Change list into a numpy array
    rvi = np.array(rvis_rice + rvis_nrice)
    times = np.array(times_rice + times_nrice)
    
    # Create a numpy array of boolean, 1 if there is rice, 0 if not
    y = np.array([1] * len(rvis_rice) + [0] * len(rvis_nrice))
    
    # Save the numpy array into a file
    np.save("rvi.npy", rvi)
    np.save("times.npy", times)
    np.save("y.npy", y)
    
def import_save_data(date, box_size_deg, resolution):
    '''
    Import RVI and date from the satelite sentinel 1 and save them in a file
    
    Args:
        date (string):        The time range between which the data is to be taken. It is in the format of "start_date/end_date" (e.g., "2021-01-01/2022-12-31")
        box_size_deg (float): The size of the bounding box to extract data from
        resolution (float):   The resolution of the satelite data
    '''
    pc.settings.set_subscription_key('**')    
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    
    import_and_save_train_data(date, box_size_deg, resolution, catalog)
    import_and_save_X_sub(date, box_size_deg, resolution, catalog)