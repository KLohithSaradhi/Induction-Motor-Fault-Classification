import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class EntireModel:

    def __init__(self, filename):
        """
        args:
            filename : name of CSV file

        Initializes the raw data after removing infinite and null values from the given file
        """
        self.filename = filename

        data = pd.read_csv(self.filename)

        X = data.select_dtypes(exclude = "int64")
        Y = data["label"]

        fft = X.select_dtypes("float64").T

        finite_X = X.drop(np.where(np.max(fft) == np.inf)[0])
        finite_Y = Y.drop(np.where(np.max(fft) == np.inf)[0])

        finite_Y = finite_Y[~finite_X.isna().any(axis = 1)]
        finite_X = finite_X[~finite_X.isna().any(axis = 1)]

        finite_X = finite_X.reset_index(drop=True)
        finite_Y = finite_Y.reset_index(drop=True)
        
        clean_X = finite_X.drop(["0", "ID"], axis = 1)

        self.X = clean_X.reset_index(drop=True)
        self.Y = np.reshape(np.array(finite_Y), (-1,1))

    def oneHotEncode(self):
        """
        Pre-splitting Function

        Instatiates a OneHotEncoder object to encode the labels
        """
        self.oneHotEncoder = OneHotEncoder()
        self.Y = self.oneHotEncoder.fit_transform(self.Y).toarray()
        
    def normalize(self, scaling = 1):
        """
        Pre-splitting Function

        args:
            scaling : The factor by which the normalized values should be scaled

        Normalizes the Input data to make each vector have unit magnitude 
        
        """
        norm = np.reshape(np.linalg.norm(self.X, axis = 1), (-1, 1))
        self.X = (self.X / norm) * scaling

    def split(self, splitIndices = None, test_size = 0.2, seed = 0 ):
        """
        args:
            test_size : The fraction of the data to be taken as test split
            seed : sets seed for random_state

        splits data into train and test splits
        instantiates a PCA object to control the explained_variance_threshold in the reduceFeatures step
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=seed, stratify=self.Y)

        self.testPCA = PCA()
        self.testPCA.fit(self.X_train)

    def reduceFeatures(self, threshold = 0.9999):
        """
        args :
            threshold : Dictates the number of PCs to be taken

        Instantiates a PCA object with a tweakable amount of PCs based on the threshold
        """
        self.pcaFeatureReducer = PCA(n_components=len(np.where(np.cumsum(self.testPCA.explained_variance_ratio_) <= threshold)[0]))
        self.X_train = self.pcaFeatureReducer.fit_transform(self.X_train)
    
    def buildKNN(self, neighbors = 5):
        """
        args :
            neighbors : The number of neighbors to be considered in KNN

        Instantiates a KNN model object trained on the train split,
        Prints the accuracy of the built mode on the test split
        
        """

        self.model = KNeighborsClassifier(n_neighbors=neighbors)
        self.model.fit(self.X_train, self.Y_train)

        print(accuracy_score(self.model.predict(self.pcaFeatureReducer.transform(self.X_test)), self.Y_test))

    def predict(self, X = None):
        """
        args : 
            X : Input data for which prediction is to be done
                default : x_test from the split step

        Returns prediction of the model
        """

        if X == None:
            X  = self.X_test

        X = self.pcaFeatureReducer.transform(X)

        return self.model.predict(X)
    
    def run(self, pipeline = {"oneHotEncode" : [1],
                                     "normalize" : [1, 1],
                                     "split" : [1, 0.2, 0],
                                     "PCA" : [1, 0.9999],
                                     "KNN" : [1, 5]}):
        """
        args:
            pipeline :
                A Dictionary that lets choose and alter steps in the pipeline
        
        Runs the set of instructions dictated by the pipeline
        """

        if pipeline["oneHotEncode"][0]:
            self.oneHotEncode()
        if pipeline["normalize"][0]:
            self.normalize(pipeline["normalize"][1])
        if pipeline["split"][0]:
            self.split(pipeline["split"][1], pipeline["split"][2])
        if pipeline["PCA"][0]:
            self.reduceFeatures(pipeline["PCA"][1])
        if pipeline["KNN"][0]:
            self.buildKNN(pipeline["KNN"][1])
        