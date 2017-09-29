import pandas as pd
import numpy as np
from scipy.spatial import distance
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.DataFrame.from_csv("heat_results.csv")

#load cities' information
with open('cities_dictionary.json') as json_data:
    cities_dict = json.load(json_data)

del cities_dict['EXAMPLE']

city_list = []
for key in cities_dict:
    city_list.append(key)
print city_list

feature_cols = ["Room type", "Accommodates", "Bedrooms", "Bathrooms", "Beds", "Number of reviews", "Review scores"]

#print feature_cols

#print city_list

#print df.head(5)

del df["Review scores"]
del df["Number of reviews"]

df.sort_index(inplace=True)

values = []
for index, row in df.iterrows():
    row = row.tolist()
    values = values + row
print values

plt.style.use('ggplot')

sns.heatmap(df, annot=False, cmap="YlGnBu", vmax = 0.35)
#vmin = 0.00000000000000000000011, vmax = 0.00000000000000011
plt.matplotlib.rcParams.update({'font.size': 34})
plt.suptitle("k = 100 heat map", fontsize=36, fontweight='bold')
rotation="vertical"

plt.rcParams["font.family"] = "cursive"
plt.tight_layout()
plt.show()


