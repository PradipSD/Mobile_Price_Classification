# The libraries & modules which we are going to use in our study:
import matplotlib
import pandas as pd
import numpy as np
from pasta.augment import inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data  = pd.read_csv('updated_train_data.csv')


data.head(10)

# id: ID
#
# battery_power: Total energy a battery can store in one time measured in mAh
#
# blue: Has bluetooth or not
#
# clock_speed: speed at which microprocessor executes instructions
#
# dual_sim: Has dual sim support or not
#
# fc: Front Camera mega pixels
#
# four_g: Has 4G or not
#
# int_memory: Internal Memory in Gigabytes
#
# m_dep: Mobile Depth in cm
#
# mobile_wt: Weight of mobile phone
#
# n_cores: Number of cores of processor
#
# pc: Primary Camera mega pixels
#
# px_height: Pixel Resolution Height
#
# px_width: Pixel Resolution Width
#
# ram: Random Access Memory in Megabytes
#
# sc_h: Screen Height of mobile in cm
#
# sc_w: Screen Width of mobile in cm
#
# talk_time: longest time that a single battery charge will last when you are
#
# three_g: Has 3G or not
#
# touch_screen: Has touch screen or not
#
# wifi: Has wifi or not
#
# price_range: This is the target variable with value of 0 (low cost), 1 (medium cost), 2 (high cost) and 3 (very high cost)

data.columns

data.shape

data.dtypes

pd.isnull(data).sum()

data.describe()

x = data.drop('price_range',axis=1)
y = data['price_range']

y.unique()

labels = ["low cost", "medium cost", "high cost", "very high cost"]
values = data['price_range'].value_counts().values
colors = ['yellow','turquoise','lightblue', 'pink']
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.set_title('balanced or imbalanced?')
plt.show()
#dataset is balanced

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101, stratify = y)

# check whether the split works correctly
print(x_train.shape)
print(x_valid.shape)

# Before going through machine learning applications, let's see the correlation btw features and target variable by plotting heatmap:
fig = plt.subplots (figsize = (12, 12))
sns.heatmap(data.corr (), square = True, cbar = True, annot = True, cmap="GnBu", annot_kws = {'size': 8})
plt.title('Correlations between Attributes')
plt.show ()

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)

y_pred_knn = model_knn.predict(x_valid)

print(metrics.confusion_matrix(y_valid, y_pred_knn))

print(accuracy_score(y_valid, y_pred_knn))

from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':np.arange(1,30)}
knn = KNeighborsClassifier()

model = GridSearchCV(knn, parameters, cv=5)
model.fit(x_train, y_train)
model.best_params_

model_knn = KNeighborsClassifier(n_neighbors=9)
model_knn.fit(x_train, y_train)

y_pred_knn = model_knn.predict(x_valid)

print(metrics.confusion_matrix(y_valid, y_pred_knn))

acc_knn = accuracy_score(y_valid, y_pred_knn)
acc_knn



