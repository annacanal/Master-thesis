from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def normalize(data):
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    # keep all features with non-zero variance, i.e. remove the features that have the same value in all samples
    feature_selector = VarianceThreshold()
    data_normalized = feature_selector.fit_transform(data_normalized)
    return data_normalized

def robust_scale_select(data):
    data_normalized = preprocessing.robust_scale(data, axis=0)
    feature_selector = VarianceThreshold()
    data_normalized = feature_selector.fit_transform(data_normalized)
    return data_normalized

def scale_select(data):
    data_normalized = preprocessing.scale(data, axis=0)
    return data_normalized



