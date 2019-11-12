import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset = pd.read_csv("pulsar_stars.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values
dataset['target_class'].value_counts()

sns.pairplot(dataset)
sns.heatmap(dataset.isnull(), cmap='viridis', yticklabels=False)
sns.heatmap(dataset.corr(), cmap='coolwarm', annot=True)
plt.tight_layout()


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

model = Sequential()

model.add(Dense(128, input_dim=8, activation='relu', init='uniform'))
model.add(Dense(64, activation='relu', init='uniform'))
model.add(Dense(32, activation='relu', init='uniform'))
model.add(Dense(16, activation='relu', init='uniform'))
model.add(Dense(1, activation='sigmoid', init='uniform'))

model.compile(metrics=['accuracy'], optimizer='adam', loss='binary_crossentropy')
model.summary()



history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save_weights("Pulsar_Star.hdf5")

y_pred = model.predict(X_val)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
