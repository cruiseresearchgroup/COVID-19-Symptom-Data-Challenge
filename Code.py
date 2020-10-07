# -*- coding: utf-8 -*-


# import required  library 
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional, TimeDistributed,Dropout
#from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical

#convert series to supervised learning sequence data based on the sliding window length
def series_to_sliding_window_sequences(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


######################################################################
# load dataset and preprocessing 
dataset = read_csv(r"C:\Users\Dell\Desktop\FL.csv", header=0, index_col=0)

# Handle empty values- if any 
dataset.fillna(dataset.mean(), inplace=True)

# get the values from the dataframe
values = dataset.values

# integer encode direction for string attribute- if any
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])

# ensure all values are float
values = values.astype('float32')

# Handle Inf values- if any
#values[values >= 1E308] = 0

# normalization of features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaledValues = scaler.fit_transform(values)

#########################################################################
sliding_window_lengh = 14 # 14 or 7 days
Number_of_predicted_days_per_widow = 1
number_of_attribute = 49
number_of_instance = 164
sliding_windows_sequances = series_to_sliding_window_sequences(values, sliding_window_lengh, Number_of_predicted_days_per_widow)

#Drop out non target attribute of the day (t) coulmns 
for i in reversed(range((number_of_attribute+1)*sliding_window_lengh, ((number_of_attribute+1)*(sliding_window_lengh+1))-1)):
    print (i)
    sliding_windows_sequances.drop(sliding_windows_sequances.columns[[i]], axis=1, inplace=True)
    
print(sliding_windows_sequances.head())

# save the new sequences  in .csv file
NewValues = sliding_windows_sequances.values
#np.savetxt(r"C:\Users\Dell\Desktop\sequances.csv", NewValues, delimiter=",")

#########################################################################
# convert cases number into relative values
targetvalues = values[:, -1] #cases number
relativetargetvalues = []
for i in range(sliding_window_lengh ,number_of_instance):
    s = 0 #sum values of the previous cases number 
    for x in range(i-sliding_window_lengh , i):
        s = s + targetvalues[x]
    average = s / sliding_window_lengh
    relativ_value = (targetvalues[i]-average)/average
    relativetargetvalues.append(relativ_value)
    
relativetargetvalues = np.array(relativetargetvalues)    
########################################################################

#data splitting 
data_splitting_method = "percantge" # or "LeavOneOnt""
if (data_splitting_method == "percantge" ):
    n_train_day = 105 #70 percente of the dataset size
    train = NewValues[:n_train_day, :]
    test = NewValues[n_train_day:, :]
    trainRelativeTarget = relativetargetvalues[:n_train_day]
    testRelativeTarget = relativetargetvalues[n_train_day:]
else:
    train = values[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90,92,93,94,96,97,98,100,101,102,104,105,106,108,109,110,112,113,114,116,117,118,120,121,122,124,125, 126,128,129,130,132,133,134,136,136,138,140,141,142,144,145,146,148,149], :]
    test = values[[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,76,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147], :]
    trainRelativeTarget = relativetargetvalues[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90,92,93,94,96,97,98,100,101,102,104,105,106,108,109,110,112,113,114,116,117,118,120,121,122,124,125, 126,128,129,130,132,133,134,136,136,138,140,141,142,144,145,146,148,149]]
    testRelativeTarget = relativetargetvalues[[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,76,71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147]]


train_X, train_y = train[:, :-1],  trainRelativeTarget
test_X, test_y = test[:, :-1], testRelativeTarget


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
########################################################################
# Building LSTM model
model = Sequential()
model.add (LSTM(113, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.01))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=72,validation_split=0.2, verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
pred_y = model.predict(test_X)

# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, pred_y))
print('Test RMSE: %.3f' % rmse)
np.savetxt(r"C:\Users\Dell\Desktop\ResultReal.csv", test_y, delimiter=",")
np.savetxt(r"C:\Users\Dell\Desktop\ResultPred.csv", pred_y, delimiter=",")