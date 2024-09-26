import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Activation, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


unimol_bace = pd.read_csv('unimol_bace_regression.csv')

bins = np.array([-0.03  , -0.0288, -0.0276, -0.0264, -0.0252, -0.024 , -0.0228,
       -0.0216, -0.0204, -0.0192, -0.018 , -0.0168, -0.0156, -0.0144,
       -0.0132, -0.012 , -0.0108, -0.0096, -0.0084, -0.0072, -0.006 ,
       -0.0048, -0.0036, -0.0024, -0.0012,  0.    ,  0.0012,  0.0024,
        0.0036,  0.0048,  0.006 ,  0.0072,  0.0084,  0.0096,  0.0108,
        0.012 ,  0.0132,  0.0144,  0.0156,  0.0168,  0.018 ,  0.0192,
        0.0204,  0.0216,  0.0228,  0.024 ,  0.0252,  0.0264,  0.0276,
        0.0288,  0.03  ])

all_feats = []

for name in range(len(unimol_bace)):
    temp = pd.read_csv('sigmaprofile_bace_csvfile/'+str(name+1)+'.csv')
    out = plt.hist(temp['sigma'],bins=bins,fill=False)
    feats = out[0]
    all_feats.append(feats)

bace_feats = np.stack(all_feats)

y = np.array(unimol_bace.iloc[:,1])

rf_model = RandomForestRegressor()
cv_scores = cross_val_score(rf_model, bace_feats, y, cv=10, scoring='neg_mean_squared_error')

print("MSE:", np.mean(cv_scores))






X_train, X_test, y_train, y_test = train_test_split(bace_feats, y, test_size=0.15, random_state=123)


input_shape = (50, 1)

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping_callback])

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

