from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import re
import string
from sklearn.preprocessing import OneHotEncoder
from madlan_data_prep import prepare_data
import pickle

path = "C:\\Users\\97250\\Desktop\\תואר הנדסה\\שנה ג' סמסטר ב\\data mining\\final_project\\output_all_students_Train_v8.xlsx"
df = pd.read_excel(path)
df = prepare_data(df)


ohe = OneHotEncoder()
ohe.fit_transform(df[['City','entranceDate']]).toarray()
f_array = ohe.fit_transform(df[['City','entranceDate']]).toarray()
f_labels = ohe.categories_
df = pd.get_dummies(df, columns=['City','entranceDate'], prefix=['City','entranceDate'])
label_list = f_labels[0].tolist()
city_num = len(label_list)
columns_num = df.shape[1]
city_df= df.iloc[:,columns_num-city_num:]
column_names = city_df.columns.tolist()
feature_columns = ['hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly','Area','floor','room_number','total_floors']
feature_columns= feature_columns + column_names


# Perform feature engineering and create the feature matrix 'X' and target vector 'y'
X = df[feature_columns]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Perform standardization on the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the Elastic Net model
elastic_net = ElasticNet()
elastic_net.fit(X_train, y_train)


# Perform 10-fold cross-validation predictions
y_pred = cross_val_predict(elastic_net, X_scaled, y, cv=10)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y, y_pred)

# Print the mean squared error
print(f"Root Mean Squared Error: {math.sqrt(mse)}")

# pickle.dump(elastic_net, open("elastic_net.pkl","wb"))


