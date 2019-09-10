

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
```

    Using TensorFlow backend.
    


```python
df = pd.read_csv('C:\\Users\\admir\\Desktop\\Churn_Modelling.csv')
```


```python
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
```


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

```


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
#Creating the Neural Network
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
```

    WARNING:tensorflow:From C:\Users\admir\Anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From C:\Users\admir\Anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/100
    8000/8000 [==============================] - 1s 179us/step - loss: 0.4790 - acc: 0.7960
    Epoch 2/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4266 - acc: 0.7960
    Epoch 3/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4205 - acc: 0.8079
    Epoch 4/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4173 - acc: 0.8267
    Epoch 5/100
    8000/8000 [==============================] - 1s 80us/step - loss: 0.4153 - acc: 0.8299
    Epoch 6/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4135 - acc: 0.8312
    Epoch 7/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4120 - acc: 0.8325
    Epoch 8/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4105 - acc: 0.8339
    Epoch 9/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4098 - acc: 0.8337
    Epoch 10/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4088 - acc: 0.8339
    Epoch 11/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4075 - acc: 0.8347
    Epoch 12/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4071 - acc: 0.8332
    Epoch 13/100
    8000/8000 [==============================] - 1s 83us/step - loss: 0.4063 - acc: 0.8352
    Epoch 14/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4062 - acc: 0.8342
    Epoch 15/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4057 - acc: 0.8340
    Epoch 16/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4056 - acc: 0.8355
    Epoch 17/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4046 - acc: 0.8339
    Epoch 18/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4045 - acc: 0.8340
    Epoch 19/100
    8000/8000 [==============================] - 1s 83us/step - loss: 0.4038 - acc: 0.8359
    Epoch 20/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4042 - acc: 0.8350
    Epoch 21/100
    8000/8000 [==============================] - 1s 79us/step - loss: 0.4036 - acc: 0.8350
    Epoch 22/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4034 - acc: 0.8345
    Epoch 23/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4031 - acc: 0.8340
    Epoch 24/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4033 - acc: 0.8346
    Epoch 25/100
    8000/8000 [==============================] - 1s 75us/step - loss: 0.4032 - acc: 0.8345
    Epoch 26/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4022 - acc: 0.8332
    Epoch 27/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4028 - acc: 0.8335
    Epoch 28/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4025 - acc: 0.8350
    Epoch 29/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4025 - acc: 0.8357
    Epoch 30/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4021 - acc: 0.8339
    Epoch 31/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4022 - acc: 0.8365
    Epoch 32/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4019 - acc: 0.8370
    Epoch 33/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4021 - acc: 0.8356
    Epoch 34/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4013 - acc: 0.8356
    Epoch 35/100
    8000/8000 [==============================] - 1s 80us/step - loss: 0.4015 - acc: 0.8357
    Epoch 36/100
    8000/8000 [==============================] - 1s 82us/step - loss: 0.4021 - acc: 0.8352
    Epoch 37/100
    8000/8000 [==============================] - 1s 79us/step - loss: 0.4017 - acc: 0.8349
    Epoch 38/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4017 - acc: 0.8350
    Epoch 39/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4014 - acc: 0.8349
    Epoch 40/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4017 - acc: 0.8342
    Epoch 41/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4010 - acc: 0.8360
    Epoch 42/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4013 - acc: 0.8337
    Epoch 43/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4014 - acc: 0.8336
    Epoch 44/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4014 - acc: 0.8357
    Epoch 45/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4017 - acc: 0.8346
    Epoch 46/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4011 - acc: 0.8355
    Epoch 47/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4016 - acc: 0.8340
    Epoch 48/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4009 - acc: 0.8357
    Epoch 49/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4009 - acc: 0.8365
    Epoch 50/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4007 - acc: 0.8354
    Epoch 51/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4013 - acc: 0.8351
    Epoch 52/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4004 - acc: 0.8355
    Epoch 53/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4008 - acc: 0.8342
    Epoch 54/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4004 - acc: 0.8349
    Epoch 55/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4007 - acc: 0.8334
    Epoch 56/100
    8000/8000 [==============================] - 1s 80us/step - loss: 0.4005 - acc: 0.8349
    Epoch 57/100
    8000/8000 [==============================] - 1s 79us/step - loss: 0.4008 - acc: 0.8350
    Epoch 58/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4007 - acc: 0.8352
    Epoch 59/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4004 - acc: 0.8346
    Epoch 60/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4007 - acc: 0.8351
    Epoch 61/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4002 - acc: 0.8357
    Epoch 62/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4003 - acc: 0.8345
    Epoch 63/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4010 - acc: 0.8349
    Epoch 64/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4003 - acc: 0.8351
    Epoch 65/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4006 - acc: 0.8347
    Epoch 66/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4003 - acc: 0.8350
    Epoch 67/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4005 - acc: 0.8350
    Epoch 68/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.3998 - acc: 0.8336
    Epoch 69/100
    8000/8000 [==============================] - 1s 79us/step - loss: 0.4008 - acc: 0.8347
    Epoch 70/100
    8000/8000 [==============================] - 1s 83us/step - loss: 0.4002 - acc: 0.8351
    Epoch 71/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4006 - acc: 0.8362
    Epoch 72/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4004 - acc: 0.8341
    Epoch 73/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4002 - acc: 0.8340
    Epoch 74/100
    8000/8000 [==============================] - 1s 76us/step - loss: 0.4004 - acc: 0.8349
    Epoch 75/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4008 - acc: 0.8341
    Epoch 76/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4004 - acc: 0.8351
    Epoch 77/100
    8000/8000 [==============================] - 1s 85us/step - loss: 0.4002 - acc: 0.8339
    Epoch 78/100
    8000/8000 [==============================] - 1s 82us/step - loss: 0.4006 - acc: 0.8346
    Epoch 79/100
    8000/8000 [==============================] - 1s 81us/step - loss: 0.4004 - acc: 0.8342: 0s - loss: 0.3800 -
    Epoch 80/100
    8000/8000 [==============================] - 1s 81us/step - loss: 0.4003 - acc: 0.8331
    Epoch 81/100
    8000/8000 [==============================] - 1s 87us/step - loss: 0.4001 - acc: 0.8374
    Epoch 82/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4005 - acc: 0.8361
    Epoch 83/100
    8000/8000 [==============================] - 1s 82us/step - loss: 0.4003 - acc: 0.8342
    Epoch 84/100
    8000/8000 [==============================] - 1s 77us/step - loss: 0.4001 - acc: 0.8340
    Epoch 85/100
    8000/8000 [==============================] - 1s 88us/step - loss: 0.3996 - acc: 0.8372
    Epoch 86/100
    8000/8000 [==============================] - 1s 78us/step - loss: 0.4004 - acc: 0.8347
    Epoch 87/100
    8000/8000 [==============================] - 1s 83us/step - loss: 0.4001 - acc: 0.8337
    Epoch 88/100
    8000/8000 [==============================] - 1s 80us/step - loss: 0.4004 - acc: 0.8344
    Epoch 89/100
    8000/8000 [==============================] - 1s 103us/step - loss: 0.4000 - acc: 0.8341
    Epoch 90/100
    8000/8000 [==============================] - 1s 129us/step - loss: 0.3998 - acc: 0.8341
    Epoch 91/100
    8000/8000 [==============================] - 1s 122us/step - loss: 0.3999 - acc: 0.8344
    Epoch 92/100
    8000/8000 [==============================] - 1s 144us/step - loss: 0.4001 - acc: 0.8345
    Epoch 93/100
    8000/8000 [==============================] - 1s 141us/step - loss: 0.4000 - acc: 0.8337
    Epoch 94/100
    8000/8000 [==============================] - 1s 134us/step - loss: 0.4003 - acc: 0.8359
    Epoch 95/100
    8000/8000 [==============================] - 1s 123us/step - loss: 0.3999 - acc: 0.8356
    Epoch 96/100
    8000/8000 [==============================] - 1s 101us/step - loss: 0.4005 - acc: 0.8345
    Epoch 97/100
    8000/8000 [==============================] - 1s 106us/step - loss: 0.4003 - acc: 0.8365
    Epoch 98/100
    8000/8000 [==============================] - 1s 102us/step - loss: 0.3998 - acc: 0.8331
    Epoch 99/100
    8000/8000 [==============================] - 1s 107us/step - loss: 0.4002 - acc: 0.8341
    Epoch 100/100
    8000/8000 [==============================] - 1s 112us/step - loss: 0.3999 - acc: 0.8327
    




    <keras.callbacks.History at 0x13d2654d940>




```python
#Let's see our accuracy. If the predict is greater than 0.5, the customer has a high chance of leaving the bank (in this scenario).
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
```


```python
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```


```python
cm
```




    array([[1542,   53],
           [ 264,  141]], dtype=int64)




```python
#Our accuracy can be estimated by adding our True Positives & Negatives and dividing by the total amount of observations:
#(1542 + 141)/2000 = 84.2%
#Not bad but we can certainly increase it with feature engineering.
```
