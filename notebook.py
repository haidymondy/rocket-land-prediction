# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/kaggle/input/spacex-launches-data/SpaceX_Launches_Data.csv')
df.head()


df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
columns_to_drop = ['FlightNumber','Date','Serial','Longitude','Latitude','Flights']

df.drop(columns=columns_to_drop, inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_columns = [column for column in df.columns if df[column].dtype.name == 'object']
for column in categorical_columns:
   df[column] = le.fit_transform(df[column])
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_columns = [column for column in df.columns if df[column].dtype.name == 'bool']
for column in categorical_columns:
   df[column] = le.fit_transform(df[column].astype(np.int))
df.info()
df = df.fillna(df.mean())
df.isnull().sum()
from sklearn.preprocessing import StandardScaler

stand_data=['PayloadMass']

scaler = StandardScaler()
df[stand_data] = scaler.fit_transform(df[stand_data])

df[stand_data].head()
import seaborn as sns
import matplotlib.pyplot as plt
#correlation heatmap of data
plt.figure(figsize = (25,10))
sns.heatmap(df.corr(), cmap="OrRd",annot=True)
X = df.drop(['Outcome'], axis=1.0)
Y = df['Outcome'] 
df['Outcome'].nunique
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test,y_test))
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy*100,"%")

import pickle

def save_model(model, filename):
    # Save the model to disk
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # Load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
filenamelr = 'model.sav'

save_model(dtc, filenamelr)
model_rndclass_loaded= load_model(filenamelr)

