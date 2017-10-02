import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt

columns = ["price", "room_type", "accommodates", "bedrooms", "bathrooms", "beds", "number_of_reviews", "review_scores_value"]

#load cities' information
with open('cities_dictionary.json') as json_data:
    cities_dict = json.load(json_data)

del cities_dict['EXAMPLE']

city_list = []
for key in cities_dict:
    city_list.append(key)
print city_list

heat_dict = {}

for city in city_list:

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

    #number reviews mean
    reviews_mean = float(city_listings["number_of_reviews"].mean())

    #average review score
    score_mean = float(city_listings["review_scores_value"].mean())

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

    N_columns = ["price", "N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_review_scores_value"]

    #drop old columns
    normal_city_listings = city_listings[N_columns]


    my_property = {"room_type": [1.0,],"accommodates": [2.0,], "bedrooms": [1.0,], "bathrooms": [1.0,], "beds": [1.0,], "number_of_reviews": [reviews_mean,], "review_scores_value": [score_mean,]}

    #"latitude": [37.98463566,], "longitude": [23.7370348,]

    my_property_df = pd.DataFrame(my_property)

    #normalise columns
    for items in columns[1:]:
        mean = city_listings[items].mean()
        N_items = "N_"+items
        std = np.std(city_listings[items])
        my_property_df[N_items] = (my_property_df[items] - mean) / std

    #drop original-value columns
    N_my_property_df=my_property_df[N_columns[1:]]

    #print city, N_my_property_df


    #Y = distance.euclidean([1,0,4], [5,0,6])
    #print("Wut", Y)
    #Y = distance.euclidean(N_my_property_df[feature_cols], normal_city_listings.iloc[0][feature_cols])
    #print("Hey",Y)

    #choose columns you want to take into account for the purpose of calculating the price
    feature_cols = ["N_room_type", "N_accommodates", "N_bedrooms", "N_bathrooms", "N_beds", "N_number_of_reviews", "N_review_scores_value"]


    N_my_property_df = N_my_property_df[feature_cols]
    selected_normal_city_listings = normal_city_listings[feature_cols]

    distance_series = selected_normal_city_listings.apply(lambda row: distance.euclidean(N_my_property_df, row),axis=1) 

    #print(distance_series)

    normal_city_listings = normal_city_listings.assign(distance=distance_series)

    normal_city_listings.sort_values("distance", inplace=True)

    #set parameter for k-nearest
    k = 5

    knn = normal_city_listings.iloc[:k]

    #print knn
   
    heat_list = []
    gap = 0
    for item in feature_cols:
        pred = knn[item].mean()
        pred = pred.item()
        val = N_my_property_df[item]
        val = val.item()
        gap = ((pred - val)**2)**0.5 
        heat_list.append(gap)

    heat_dict[city] = heat_list



feature_cols = ["Room type", "Accommodates", "Bedrooms", "Bathrooms", "Beds", "Number of reviews", "Review scores"]

df=pd.DataFrame.from_items(heat_dict.iteritems(), 
                            orient='index', 
                            columns=feature_cols)

print df


df.to_csv('heat_results.csv', index=True)
