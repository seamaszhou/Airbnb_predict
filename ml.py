import pandas as pd
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings.head()

our_acc_value = 3
first_living_space_value = dc_listings.loc[0,'accommodates']

first_distance = np.abs(first_living_space_value - our_acc_value)


dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)
dc_listings.distance.value_counts().sort_index()

dc_listings = dc_listings.sample(frac=1, random_state=0)
dc_listings = dc_listings.sort_values('distance')
dc_listings.price.head()

dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)
mean_price = dc_listings.price.iloc[:5].mean()

dc_listings.drop('distance', axis=1)
train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)
test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column='accommodates')

test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (0.5)
print rmse

for feature in ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']:
    test_df['predicted_price'] = test_df.accommodates.apply(predict_price, feature_column=feature)
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
    mse = test_df['squared_error'].mean()
    rmse = mse**(0.5)
    print("RMSE for the {} column:{}".format(feature,rmse))
