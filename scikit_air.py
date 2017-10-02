from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json
import math
from sklearn.metrics import mean_squared_error

columns = ["price", "room_type", "accommodates", "bedrooms", "bathrooms", "beds", "number_of_reviews", "latitude", "longitude", "review_scores_value"]

#load cities' information
with open('cities_dictionary.json') as json_data:
    cities_dict = json.load(json_data)

del cities_dict['EXAMPLE']

#choose the city
city = "AMSTERDAM"

#upload data
try:
    city_listings = pd.read_csv("DATA/raw/" + city + "_listings.csv")
except Exception:
    if city == "HONG KONG":
        city = "HK"
        city_listings = pd.read_csv("DATA/raw/" + city + "_listings.csv")
        city = "HONG KONG"
    if city == "LOS ANGELES":
        city = "LA"
        city_listings = pd.read_csv("DATA/raw/" + city + "_listings.csv")
        city = "LOS ANGELES"
    if city == "SAN FRANCISCO":
        city = "SF"
        city_listings = pd.read_csv("DATA/raw/" + city + "_listings.csv")
        city = "SAN FRANCISCO"

#select relevant columns from the data
city_listings = city_listings[columns]

#drop room types that are not well formatted
TF = (city_listings["room_type"] == "Entire home/apt") | (city_listings["room_type"] == "Private room")
city_listings = city_listings[TF]
#drop NaN rows, which means we mostly drop items which have no reviews
city_listings = city_listings.dropna()
#shuffle
city_listings = city_listings.sample(frac=1,random_state=0)

#remove unwanted sharacters
city_listings['price'] = city_listings.price.str.replace("\$|,",'').astype(float)
#set private room type to 0 and entire home/apt to 1
city_listings['room_type'].replace("Private room", 0.0,inplace=True)
city_listings['room_type'].replace("Entire home/apt", 1.0,inplace=True)

mean_price = city_listings["price"].mean()

#we normalise
for items in columns[1:]:
    mean = city_listings[items].mean()
    std = np.std(city_listings[items])
    N_items = "N_"+items
    city_listings[N_items] = (city_listings[items] - mean) / std

N_columns = ["price", "N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_latitude", "N_longitude", "N_review_scores_value"]

#drop old columns
normal_city_listings = city_listings[N_columns]

split_value = int(round(float(city_listings.shape[0])*75/100))

#we use 75% of the dataset as train, 25% as test

train_set = normal_city_listings.iloc[:split_value]
test_set = normal_city_listings.iloc[split_value:]


#choose columns you want to take into account for the purpose of calculating the price
feature_cols = ["N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_latitude", "N_longitude", "N_review_scores_value", "N_number_of_reviews"]

#print train_set


#create an instance of KNeighborsRegressor
knn = KNeighborsRegressor(algorithm='brute')

#store the training set, specify columns to use to predict prices and the taget (price)
knn.fit(train_set[feature_cols], train_set["price"])

#make price prediction for all feature columns, this hands back a numpy 1-d array with predicted price for each test listing
predictions = knn.predict(test_set[feature_cols])

predictions = predictions.tolist()
print predictions[:10]
test_set_price = test_set["price"].tolist()
print test_set_price[:10]

#calculate mean square error

features_mse = mean_squared_error(test_set_price, predictions)
features_rmse = features_mse**(0.5)
print features_rmse










