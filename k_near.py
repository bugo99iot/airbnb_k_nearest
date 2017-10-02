import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json

columns = ["price", "room_type", "accommodates", "bedrooms", "bathrooms", "beds", "number_of_reviews", "latitude", "longitude", "review_scores_rating"]

#load cities' information
with open('cities_dictionary.json') as json_data:
    cities_dict = json.load(json_data)

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
#we normalise
for items in columns[1:]:
    mean = city_listings[items].mean()
    std = np.std(city_listings[items])
    N_items = "N_"+items
    city_listings[N_items] = (city_listings[items] - mean) / std

N_columns = ["price", "N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_latitude", "N_longitude", "N_review_scores_rating"]

#drop old columns
normal_city_listings = city_listings[N_columns]


my_property = {"room_type": [0.0,],"accommodates": [2.0,], "bedrooms": [1.0,], "bathrooms": [1.0,], "beds": [1.0,], "number_of_reviews": [135.0,], "latitude": [37.98463566,], "longitude": [23.7370348,], "review_scores_rating": [95.0,]}

#"latitude": [37.98463566,], "longitude": [23.7370348,]

my_property_df = pd.DataFrame(my_property)

#will need this later on to get real latitude and longitude
long_avg = city_listings["longitude"].mean()
long_std = np.std(city_listings["longitude"])
lat_avg =city_listings["latitude"].mean()
lat_std =np.std(city_listings["latitude"])

#normalise columns
for items in columns[1:]:
    mean = city_listings[items].mean()
    N_items = "N_"+items
    std = np.std(city_listings[items])
    my_property_df[N_items] = (my_property_df[items] - mean) / std

#drop original-value columns
N_my_property_df=my_property_df[N_columns[1:]]


#choose columns you want to take into account for the purpose of calculating the price
feature_cols = ["N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_review_scores_rating", "N_latitude", "N_longitude"]


def predict_price_multivariate(row, test_item):
    distance_series = distance.euclidean(test_item, row)
    return(distance_series)


N_my_property_df = N_my_property_df[feature_cols]
selected_normal_city_listings = normal_city_listings[feature_cols]

distance_series = selected_normal_city_listings.apply(predict_price_multivariate, test_item=N_my_property_df, axis=1) 

#print(distance_series)

normal_city_listings = normal_city_listings.assign(distance=distance_series)

normal_city_listings.sort_values("distance", inplace=True)

if "N_longitude" in feature_cols and "N_latitude" in feature_cols:
    normal_city_listings["longitude"] = (normal_city_listings["N_longitude"]*long_std)+long_avg
    normal_city_listings["latitude"] = (normal_city_listings["N_latitude"]*lat_std)+lat_avg

#set parameter for k-nearest
k = 6

knn = normal_city_listings.iloc[:k]
print knn

predicted_price = knn["price"].mean()

#adjust for the conversion rate
predicted_price = int(round(predicted_price*cities_dict[city][0]))


print "The price for a 1 night stay in", city, "is", predicted_price, "euro."
print "In 1 month a host in", city, "typically makes", int(round(predicted_price*30*cities_dict[city][1])), "euro."
print "Monthly rent in", city, "is typically", int(round(cities_dict[city][2])),"euro."

