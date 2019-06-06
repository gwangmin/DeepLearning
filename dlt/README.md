# Deep Learning Toolkit v1.1.1
This toolkit provides Neural Network.

  
  
  
  
## Required packages
* tensorflow
* keras
* numpy
* matplotlib

## Contents
### dlt.dnn
* DNNClassifier(units list) : DNN(Deep Neural Network) Classifier
* DNNRegressor(units list) : DNN Regressor
### dlt.plot
* showLoss(history or history callback) : show loss graph
* showAcc(history or history callback) : show accuracy graph
* visualizeFilter(filters, grid size): visualize specified filters
### dlt.callbacks
* HistoryCallback(): keras callback for graph above. It must be called init()


## Tips
### preprocessing
* must use numpy.array
* use keras.utils.to_categorical() for one hot
### fit method
* use validation_data or validation_split arg for validation
* use EarlyStopping() callback for early stopping
### visualize
* use dlt.plot or use tensorboard
### keras.datasets(https://keras.io/datasets/)
* for using mnist, follow these
    (x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
    reshaping, type casting, normalization the x data
	convert to one hot label


