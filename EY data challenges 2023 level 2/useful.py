import numpy as np
import pandas as pd
import statsmodels.api as sm
import pystac
import odc
import pystac_client
from odc.stac import stac_load
from pystac_client import Client
import planetary_computer as pc
from sklearn.metrics import r2_score
from ast import literal_eval
from tqdm.notebook import tqdm
from scipy.stats import kurtosis, skew, variation
import statsmodels.api as sm

def get_submission_multiple_regressor_scaler(regressors, data_name, scs, to_drop):
    """
    Generates predictions for rice crop yield using multiple regressor models and scaling the input data.

    Parameters:
    -----------
    regressors : list of sklearn regressor models
        The regressor models to use for making predictions.
    data_name : list of str
        The names of the files containing the submission data to use for making predictions.
    scs : list of sklearn StandardScaler
        The StandardScaler objects used to scale the input data for each regressor.

    Returns:
    --------
    None

    This function generates predictions for rice crop yield using multiple regressor models, scaling the input data using
    a StandardScaler object for each regressor. It assumes that the input data files are located in the data/ directory
    and are named "submission_data_{name}.csv", where {name} is one of the names in the `data_name` list.
    
    The function reads in the submission data files and calculates statistical features for each row of each file using
    the `generate_statistical_features_v4` function. It then scales the resulting feature arrays using the corresponding
    StandardScaler object for each regressor model. The scaled feature arrays are then concatenated together and used to
    make predictions using each regressor model. The final prediction for each row is the average of the predictions from
    all regressor models.
    
    The predictions are combined into a Pandas DataFrame and saved as a CSV file named
    "challenge_2_submission_rice_crop_yield_prediction.csv".
    """
    test_file = pd.read_csv('Challenge_2_submission_template.csv')
    sub_sat_datas = []
    for name in data_name:
        sub_sat_datas.append(pd.read_csv("data/submission_data_" + name + ".csv").drop(to_drop, axis=1))

    for col in sub_sat_datas[0].columns:
        for sat_data in sub_sat_datas:
            sat_data[col] = sat_data[col].str.replace(', nan', '')
            sat_data[col] = sat_data[col].str.replace('nan, ', '')
            sat_data[col] = sat_data[col].apply(literal_eval)

    sub_features = []
    for sat_data in sub_sat_datas:
        sub_features.append(np.array(generate_statistical_features_v4(sat_data)))
    
    X_sub = np.concatenate(sub_features, axis = 1)
    
    #Making predictions
    preds = []
    for reg, sc in zip(regressors, scs):
        preds.append(reg.predict(sc.transform(X_sub)))
    final_predictions = np.mean(preds, axis=0)
    final_prediction_series = pd.Series(final_predictions)
    #Combining the results into dataframe
    test_file['Predicted Rice Yield (kg/ha)']=list(final_prediction_series)
    #Dumping the predictions into a csv file.
    test_file.to_csv("challenge_2_submission_rice_crop_yield_prediction.csv",index = False)

def generate_statistical_features_v4(dataframe):
    """
    Calculates statistical features for each row of a Pandas DataFrame containing multiple time series.

    Parameters:
    -----------
    dataframe : Pandas DataFrame
        The input DataFrame, where each column represents a time series.

    Returns:
    --------
    features_list : list of arrays
        A list of arrays, where each array represents the statistical features for a single row of the input DataFrame.
        The arrays are padded with zeros for missing values or NaNs.
    
    Statistical Features:
    ---------------------
    For each column in the input DataFrame, the following statistical features are calculated:
    - Minimum value
    - Maximum value
    - Range (difference between maximum and minimum values)
    - Mean
    - Median
    - Standard deviation
    - Variance
    - Skewness
    - Kurtosis
    - 25th percentile
    - 75th percentile
    - Autocorrelation at lag 1 (if the time series has more than 1 value)
    - Partial autocorrelation at lag 1 (if the time series has more than 2 values)
    - Permutation entropy

    The function uses the `numpy`, `scipy`, and `pyentrp` libraries to calculate some of these statistical features.
    The `tqdm` library is used to display a progress bar during the iteration over rows of the DataFrame.
    """
    features_list = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        tempo_list = []
        for i in range(row.shape[0]):
            tempo_list.append(min(row[i]))
            tempo_list.append(max(row[i]))
            tempo_list.append(max(row[i]) - min(row[i]))
            tempo_list.append(np.mean(row[i]))
            tempo_list.append(np.median(row[i]))
            tempo_list.append(np.std(row[i]))
            tempo_list.append(np.var(row[i]))
            tempo_list.append(skew(row[i]))
            tempo_list.append(kurtosis(row[i]))
            tempo_list.append(np.percentile(row[i], 25))
            tempo_list.append(np.percentile(row[i], 75))
            if len(row[i]) > 1:
                tempo_list.append(sm.tsa.acf(row[i], fft=True, nlags=1)[1])
                if len(row[i]) > 2:
                    tempo_list.append(sm.tsa.pacf(row[i])[0])
                else:
                    tempo_list.append(0)
            else:
                tempo_list.append(0)
                tempo_list.append(0)
            tempo_list.append(permutation_entropy(row[i], dx=min(6, len(row[i])), base=2, normalized=True))

        features_list.append(np.nan_to_num(tempo_list))
    return features_list


def get_score(model, X_train, y_train, X_test, y_test, print_score=True):
    """
    Calculates the R2 score for a given regression model using training and test data.

    Parameters:
    -----------
    model : sklearn regressor
        The regressor model to use for making predictions.
    X_train : numpy array
        The training input features.
    y_train : numpy array
        The training target values.
    X_test : numpy array
        The test input features.
    y_test : numpy array
        The test target values.
    print_score : bool
        A flag indicating whether to print the R2 scores.

    Returns:
    --------
    insample_r2 : float
        The R2 score for the model on the training data.
    outsample_r2 : float
        The R2 score for the model on the test data.

    This function calculates the R2 score for a given regression model using training and test data. The R2 score is a
    measure of how well the model fits the data, with higher values indicating a better fit. The function uses the
    `predict` method of the input `model` to make predictions for both the training and test data, and then calculates
    the R2 score using the `r2_score` function from the `sklearn.metrics` module.

    If the `print_score` flag is True (default), the function prints the R2 scores for the training and test data.

    The function returns the R2 scores for both the training and test data as a tuple of two floats.
    """
    insample_predictions = model.predict(X_train)
    outsample_predictions = model.predict(X_test)
    if print_score:
        print("Insample R2 Score: {0:.2f}".format(r2_score(y_train,insample_predictions)))
        print("Outsample R2 Score: {0:.2f}".format(r2_score(y_test,outsample_predictions)))
    return r2_score(y_train,insample_predictions), r2_score(y_test,outsample_predictions)

def ordinal_distribution(data, dx=3, dy=1, taux=1, tauy=1, return_missing=False, tie_precision=None):
    '''
    Returns
    -------
     : tuple
       Tuple containing two arrays, one with the ordinal patterns occurring in data 
       and another with their corresponding probabilities.
       
    Attributes
    ---------
    data : array 
           Array object in the format :math:`[x_{1}, x_{2}, x_{3}, \\ldots ,x_{n}]`
           or  :math:`[[x_{11}, x_{12}, x_{13}, \\ldots, x_{1m}],
           \\ldots, [x_{n1}, x_{n2}, x_{n3}, \\ldots, x_{nm}]]`.
    dx : int
         Embedding dimension (horizontal axis) (default: 3).
    dy : int
         Embedding dimension (vertical axis); it must be 1 for time series 
         (default: 1).
    taux : int
           Embedding delay (horizontal axis) (default: 1).
    tauy : int
           Embedding delay (vertical axis) (default: 1).
    return_missing: boolean
                    If `True`, it returns ordinal patterns not appearing in the 
                    symbolic sequence obtained from **data** are shown. If `False`,
                    these missing patterns (permutations) are omitted 
                    (default: `False`).
    tie_precision : int
                    If not `None`, **data** is rounded with `tie_precision`
                    number of decimals (default: `None`).
   
    '''
    def setdiff(a, b):
        '''
        Returns
        -------
        : array
            An array containing the elements in `a` that are not contained in `b`.
            
        Parameters
        ----------    
        a : tuples, lists or arrays
            Array in the format :math:`[[x_{21}, x_{22}, x_{23}, \\ldots, x_{2m}], 
            \\ldots, [x_{n1}, x_{n2}, x_{n3}, ..., x_{nm}]]`.
        b : tuples, lists or arrays
            Array in the format :math:`[[x_{21}, x_{22}, x_{23}, \\ldots, x_{2m}], 
            \\ldots, [x_{n1}, x_{n2}, x_{n3}, ..., x_{nm}]]`.
        '''

        a = np.asarray(a).astype('int64')
        b = np.asarray(b).astype('int64')

        _, ncols = a.shape

        dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [a.dtype]}

        C = np.setdiff1d(a.view(dtype), b.view(dtype))
        C = C.view(a.dtype).reshape(-1, ncols)

        return(C)

    try:
        ny, nx = np.shape(data)
        data   = np.array(data)
    except:
        nx     = np.shape(data)[0]
        ny     = 1
        data   = np.array([data])

    if tie_precision is not None:
        data = np.round(data, tie_precision)

    partitions = np.concatenate(
        [
            [np.concatenate(data[j:j+dy*tauy:tauy,i:i+dx*taux:taux]) for i in range(nx-(dx-1)*taux)] 
            for j in range(ny-(dy-1)*tauy)
        ]
    )

    symbols = np.apply_along_axis(np.argsort, 1, partitions)
    symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)

    probabilities = symbols_count/len(partitions)

    if return_missing==False:
        return symbols, probabilities
    
    else:
        all_symbols   = list(map(list,list(itertools.permutations(np.arange(dx*dy)))))
        miss_symbols  = setdiff(all_symbols, symbols)
        symbols       = np.concatenate((symbols, miss_symbols))
        probabilities = np.concatenate((probabilities, np.zeros(miss_symbols.__len__())))
        
        return symbols, probabilities

def permutation_entropy(data, dx=3, dy=1, taux=1, tauy=1, base=2, normalized=True, probs=False, tie_precision=None):
    '''
    Returns Permutation Entropy
    Attributes:
    data : array
           Array object in the format :math:`[x_{1}, x_{2}, x_{3}, \\ldots ,x_{n}]`
           or  :math:`[[x_{11}, x_{12}, x_{13}, \\ldots, x_{1m}],
           \\ldots, [x_{n1}, x_{n2}, x_{n3}, \\ldots, x_{nm}]]`
           or an ordinal probability distribution (such as the ones returned by :func:`ordpy.ordinal_distribution`).
    dx :   int
           Embedding dimension (horizontal axis) (default: 3).
    dy :   int
           Embedding dimension (vertical axis); it must be 1 for time series (default: 1).
    taux : int
           Embedding delay (horizontal axis) (default: 1).
    tauy : int
           Embedding delay (vertical axis) (default: 1).
    base : str, int
           Logarithm base in Shannon's entropy. Either 'e' or 2 (default: 2).
    normalized: boolean
                If `True`, permutation entropy is normalized by its maximum value 
                (default: `True`). If `False`, it is not.
    probs : boolean
            If `True`, assumes **data** is an ordinal probability distribution. If 
            `False`, **data** is expected to be a one- or two-dimensional 
            array (default: `False`). 
    tie_precision : int
                    If not `None`, **data** is rounded with `tie_precision`
                    number of decimals (default: `None`).
    '''
    if not probs:
        _, probabilities = ordinal_distribution(data, dx, dy, taux, tauy, return_missing=False, tie_precision=tie_precision)
    else:
        probabilities = np.asarray(data)
        probabilities = probabilities[probabilities>0]

    if normalized==True and base in [2, '2']:        
        smax = np.log2(float(np.math.factorial(dx*dy)))
        s    = -np.sum(probabilities*np.log2(probabilities))
        return s/smax
         
    elif normalized==True and base=='e':        
        smax = np.log(float(np.math.factorial(dx*dy)))
        s    = -np.sum(probabilities*np.log(probabilities))
        return s/smax
    
    elif normalized==False and base in [2, '2']:
        return -np.sum(probabilities*np.log2(probabilities))
    else:
        return -np.sum(probabilities*np.log(probabilities))