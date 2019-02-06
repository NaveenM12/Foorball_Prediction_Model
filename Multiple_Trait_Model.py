import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

train_df = pd.read_csv('data/Master.csv')

# print(train_df.head())

train_X = train_df.drop(columns=['Pt_Diff', 'Rivalry'])

#print(train_X.head())

train_y = train_df[['Pt_Diff', 'Rivalry']]

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
model.add(Dense(2))


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(train_X, train_y, validation_split=0.5, epochs=100)

test_X = pd.read_csv('data/TestData_Multiple.csv')

test_y_predictions = model.predict(test_X)

print(test_y_predictions)
