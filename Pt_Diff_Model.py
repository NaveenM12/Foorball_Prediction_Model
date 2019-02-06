import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

train_df = pd.read_csv('Football CSV Stats/Master.csv')

# print(train_df.head())

train_X = train_df.drop(columns=['Pt_Diff'])

#print(train_X.head())

train_y = train_df[['Pt_Diff']]

#print(train_y.head())

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(3000, activation='relu', input_shape=(n_cols,)))
model.add(Dense(3000, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(3000, activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_monitor = EarlyStopping(patience=5)

model.fit(train_X, train_y, validation_split=0.3, epochs=100, callbacks=[early_stopping_monitor])

test_X = pd.read_csv('Football CSV Stats/TestData.csv')

test_y_predictions = model.predict(test_X)

print(test_y_predictions)
