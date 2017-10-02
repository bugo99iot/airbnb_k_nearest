import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json
import math

columns = ["price", "room_type", "accommodates", "bedrooms", "bathrooms", "beds", "number_of_reviews", "latitude", "longitude", "review_scores_value"]

#load cities' information
with open('cities_dictionary.json') as json_data:
    cities_dict = json.load(json_data)

del cities_dict['EXAMPLE']

#choose the city
city = "ATHENS"

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

split_value = int(round(float(city_listings.shape[0])*75/100))

#we use 75% of the dataset as train, 25% as test
train_set = city_listings.iloc[:split_value]
test_set = city_listings.iloc[split_value:]

print test_set.head(5)

#we normalise
for items in columns[1:]:
    mean = city_listings[items].mean()
    std = np.std(city_listings[items])
    N_items = "N_"+items
    city_listings[N_items] = (city_listings[items] - mean) / std

N_columns = ["price", "N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_latitude", "N_longitude", "N_review_scores_value"]

#drop old columns
normal_city_listings = city_listings[N_columns]

train_set = normal_city_listings.iloc[:2888]
test_set = normal_city_listings.iloc[2888:]


#choose columns you want to take into account for the purpose of calculating the price
feature_cols = ["N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_latitude", "N_longitude", "N_review_scores_value", "N_number_of_reviews"]

train_set_f = train_set[feature_cols]
test_set_f = test_set[feature_cols]

standard_deviation = 0

k = 5

aces = 0

differences_squared = []

precision = 0.30

for index, rows in test_set_f.iterrows():

    distance_series = train_set_f.apply(lambda row: distance.euclidean(rows, row), axis=1) 

    train_set = train_set.assign(distance=distance_series)

    train_set.sort_values("distance", inplace=True)

    knn = train_set.iloc[:k]

    predicted_price = knn["price"].mean()
    predicted_price = predicted_price.item()

    real_price = test_set.loc[[index], :]["price"]
    real_price = real_price.item()

    differences_squared.append((predicted_price - real_price)**2)

    if predicted_price/real_price < 1 + precision and predicted_price/real_price > 1 - precision:
        aces += 1
    del train_set["distance"]

average_deviation  = sum(differences_squared) / float(len(differences_squared))
print "Aces: ", aces
rmse = (average_deviation)**0.5
print "Rmse: ", rmse, "for a price mean: ", mean_price
acespercent = float(aces)/float(test_set.shape[0])
print "Accuracy %:", acespercent, "with a precision: ", precision



