# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:49:17 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df_taxi = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Foundational ML Algorithms I/Tex6vcCLS5KedfKdbe8w_nyc_taxi_final/nyc_taxi_trip_duration.csv')
df_taxi.columns
df_taxi.dtypes
df_taxi.shape
df_taxi.head()

missing_values = df_taxi.isnull().sum()
print(missing_values)

#1 Dealing with IQR using outlier
num_cols = df_taxi.select_dtypes(include = ['int', 'float'])
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1
outliers = ((num_cols < Q1 - 1.5 * IQR | (num_cols > (Q3 + 1.5 * IQR))))
outliers_summary = outliers.sum()
print(outliers_summary)

df_taxi['pickup_datetime'] = pd.to_datetime(df_taxi['pickup_datetime'])
df_taxi['dropoff_datetime'] = pd.to_datetime(df_taxi['dropoff_datetime'])
df_taxi['hour'] = df_taxi['pickup_datetime'].dt.hour
df_taxi['day'] = df_taxi['pickup_datetime'].dt.day
df_taxi['month'] = df_taxi['pickup_datetime'].dt.month
df_taxi['day_of_week'] = df_taxi['pickup_datetime'].dt.dayofweek
df_taxi['store_and_fwd_flag'] = df_taxi['store_and_fwd_flag'].map({'Y': 1, 'N': 0})


df_taxi['trip_duration(mins)'] = df_taxi['trip_duration'] / 60
df_taxi['trip_duration(mins)'].describe()


# Calculate geodesic distance (in km) between pickup and dropoff
from geopy.distance import geodesic

def calculate_distance(row):
    return geodesic(
        (row['pickup_latitude'], row['pickup_longitude']),
        (row['dropoff_latitude'], row['dropoff_longitude'])
    ).km

df_taxi['distance_km'] = df_taxi.apply(calculate_distance, axis=1)
df_taxi['distance_km'].describe()

#Remove outliers in trip distance and duration

# Calculate IQR for trip distance
Q1d = df_taxi['distance_km'].quantile(0.25)
Q3d = df_taxi['distance_km'].quantile(0.75)
IQRd = Q3d - Q1d

# Define lower and upper bounds
lower_bound_d = Q1d - 1.5 * IQRd
upper_bound_d = Q3d + 1.5 * IQRd

# Calculate IQR for trip duration
Q1t = df_taxi['trip_duration(mins)'].quantile(0.25)
Q3t = df_taxi['trip_duration(mins)'].quantile(0.75)
IQRt = Q3t - Q1t

# Define lower and upper bounds
lower_bound = Q1t - 1.5 * IQRt
upper_bound = Q3t + 1.5 * IQRt

# Filter the data to remove outliers
filtered_data = df_taxi[(df_taxi['distance_km'] >= lower_bound_d) &
                            (df_taxi['distance_km'] <= upper_bound_d) &
                            (df_taxi['trip_duration(mins)'] >= lower_bound) &
                             (df_taxi['trip_duration(mins)'] <= upper_bound)]

# Check the shape of the filtered data
print(f"Original data size: {df_taxi.shape[0]}")
print(f"Filtered data size: {filtered_data.shape[0]}")

#  Visualize trip distance distribution
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['distance_km'], bins=50, kde=True)
plt.title("Distribution of Trip Distance (KMs)")
plt.xlabel("Trip Distance (KMs)")
plt.ylabel("Frequency")
plt.show()

# Plot the distribution of trip duration after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['trip_duration(mins)'], bins=50, kde=True)
plt.title("Distribution of Trip Duration (Minutes) - Outliers Removed")
plt.xlabel("Trip Duration (Minutes)")
plt.ylabel("Frequency")
plt.show()

# Plot trip duration by pickup hour after removing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='trip_duration(mins)', data=filtered_data)
plt.title("Trip Duration by Pickup Hour - Outliers Removed")
plt.xlabel("Pickup Hour")
plt.ylabel("Trip Duration (Minutes)")
plt.show()

# Trip duration by day of the week (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='trip_duration(mins)', data=filtered_data)
plt.title("Trip Duration by Day of the Week")
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Trip Duration (Minutes)")
plt.show()

# Relationship between distance and trip duration (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_km', y='trip_duration(mins)', data=filtered_data, alpha=0.3)
plt.title("Distance vs. Trip Duration")
plt.xlabel("Distance (km)")
plt.ylabel("Trip Duration (Minutes)")
plt.show()

# More analysis on outliers
# Long duration for short distances
long_duration_threshold = filtered_data['trip_duration(mins)'].quantile(0.95)  # Top 5% durations
short_distance_threshold = filtered_data['distance_km'].quantile(0.25)        # Bottom 25% distances

# Short duration for long distances
short_duration_threshold = filtered_data['trip_duration(mins)'].quantile(0.05)  # Bottom 5% durations
long_distance_threshold = filtered_data['distance_km'].quantile(0.75)          # Top 25% distances

# Identify outliers
long_duration_outliers = filtered_data[
    (filtered_data['distance_km'] <= short_distance_threshold) &
    (filtered_data['trip_duration(mins)'] >= long_duration_threshold)
]

short_duration_outliers = filtered_data[
    (filtered_data['distance_km'] >= long_distance_threshold) &
    (filtered_data['trip_duration(mins)'] <= short_duration_threshold)
]

# Combine outliers
all_outliers = pd.concat([long_duration_outliers, short_duration_outliers])

# Summary of outliers
print("Summary of Long Duration Outliers (Short Distance):")
print(long_duration_outliers[['distance_km', 'trip_duration(mins)']].describe())

print("\nSummary of Short Duration Outliers (Long Distance):")
print(short_duration_outliers[['distance_km', 'trip_duration(mins)']].describe())

# Visualize outliers in the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_km', y='trip_duration(mins)', data=filtered_data, alpha=0.3, label='Regular Trips')
sns.scatterplot(x='distance_km', y='trip_duration(mins)', data=long_duration_outliers, color='red', label='Long Duration Outliers')
sns.scatterplot(x='distance_km', y='trip_duration(mins)', data=short_duration_outliers, color='green', label='Short Duration Outliers')
plt.title("Distance vs. Trip Duration with Outliers Highlighted")
plt.xlabel("Distance (km)")
plt.ylabel("Trip Duration (Minutes)")
plt.legend()
plt.show()


# Distance traveled by pickup hour (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='distance_km', data=filtered_data)
plt.title("Distance Travelled by Pickup Hour")
plt.xlabel("Pickup Hour")
plt.ylabel("Distance Travelled (km)")
plt.show()


# Correlation analysis by time windows
# Define time windows
def categorize_time(hour):
    if 8 <= hour <= 10:  # Morning peak hours
        return 'Morning Peak'
    elif 17 <= hour <= 19:  # Evening peak hours
        return 'Evening Peak'
    elif 0 <= hour <= 5:  
        return 'Late Night'
    else:
        return 'Off-Peak'

# Add a time window category to the dataset
filtered_data['time_window'] = filtered_data['hour'].apply(categorize_time)

# Calculate correlation for each time window
correlation_results = {}
for time_window, group in filtered_data.groupby('time_window'):
    correlation = group['distance_km'].corr(group['trip_duration(mins)'])
    correlation_results[time_window] = correlation

# Display the correlations
print("Correlation Between Distance and Trip Duration by Time Window:")
for time_window, correlation in correlation_results.items():
    print(f"{time_window}: {correlation:.2f}")

# Visualize the relationship for each time window
plt.figure(figsize=(15, 10))
time_windows = ['Morning Peak', 'Evening Peak', 'Late Night', 'Off-Peak']
for i, time_window in enumerate(time_windows, 1):
    plt.subplot(2, 2, i)
    group = filtered_data[filtered_data['time_window'] == time_window]
    sns.scatterplot(x='distance_km', y='trip_duration(mins)', data=group, alpha=0.5)
    plt.title(f"Distance vs. Trip Duration ({time_window})")
    plt.xlabel("Distance (km)")
    plt.ylabel("Trip Duration (mins)")
plt.tight_layout()
plt.show()


# Joint distribution of distance and trip duration
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=filtered_data,
    x="distance_km", y="trip_duration(mins)",
    fill=True, cmap="Blues", levels=50, thresh=0.1
)
plt.title("Joint Distribution of Distance and Trip Duration")
plt.xlabel("Distance (km)")
plt.ylabel("Trip Duration (Minutes)")
plt.show()

# Average trip duration and distance by pickup hour
hourly_summary = filtered_data.groupby('hour')[['trip_duration(mins)', 'distance_km']].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='trip_duration(mins)', data=hourly_summary, label='Avg Trip Duration (Minutes)')
sns.lineplot(x='hour', y='distance_km', data=hourly_summary, label='Avg Distance (km)')
plt.title("Average Trip Duration and Distance by Pickup Hour")
plt.xlabel("Pickup Hour")
plt.ylabel("Average Values")
plt.legend()
plt.show()


#####################################################
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

filtered_data = filtered_data.drop(columns=['id'])


# Define features (X) and target (y)
X = filtered_data.drop(columns=['trip_duration', 'trip_duration(mins)'])
y = filtered_data['trip_duration(mins)'] 

X.columns
X = X.drop(columns=['pickup_datetime', 'dropoff_datetime','time_window' ])
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize the data for linear regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Base Linear Regression (No Regularization)
base_model = LinearRegression()
base_model.fit(X_train_scaled, y_train)
y_pred_base = base_model.predict(X_test_scaled)

# Evaluate Base Model
base_mse = mean_squared_error(y_test, y_pred_base)
base_r2 = r2_score(y_test, y_pred_base)
print("Base Regression (No Regularization) Results:")
print(f"Mean Squared Error: {base_mse}")
print(f"R-squared: {base_r2}\n")

# Define alphas to test
alphas = np.linspace(0.1,20,50)

# Function to perform cross-validation
def cross_validate_model(model, X, y, alphas):
    results = []
    for alpha in alphas:
        model.set_params(alpha=alpha)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        results.append({
            'alpha': alpha,
            'mean_mse': -np.mean(scores),
            'std_mse': np.std(scores)
        })
    return results

# Ridge Regression (L2 Regularization)
ridge = Ridge()
ridge_results = cross_validate_model(ridge, X_train_scaled, y_train, alphas)

# Lasso Regression (L1 Regularization)
lasso = Lasso(max_iter=10000)
lasso_results = cross_validate_model(lasso, X_train_scaled, y_train, alphas)

# Elastic Net (L1 + L2 Regularization)
elastic_net = ElasticNet(max_iter=10000, l1_ratio=0.5)
elastic_net_results = cross_validate_model(elastic_net, X_train_scaled, y_train, alphas)

# Display results
print("Ridge Results:")
print(pd.DataFrame(ridge_results))
print("\nLasso Results:")
print(pd.DataFrame(lasso_results))
print("\nElastic Net Results:")
print(pd.DataFrame(elastic_net_results))

# Select the best alpha for Ridge
best_ridge_alpha = min(ridge_results, key=lambda x: x['mean_mse'])['alpha']
ridge.set_params(alpha=best_ridge_alpha)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
print(f"\nBest Ridge Alpha: {best_ridge_alpha}")
print(f"Ridge Test MSE: {ridge_mse}, R2: {ridge_r2}")

# Select the best alpha for Lasso
best_lasso_alpha = min(lasso_results, key=lambda x: x['mean_mse'])['alpha']
lasso.set_params(alpha=best_lasso_alpha)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
print(f"\nBest Lasso Alpha: {best_lasso_alpha}")
print(f"Lasso Test MSE: {lasso_mse}, R2: {lasso_r2}")

# Select the best alpha for Elastic Net
best_en_alpha = min(elastic_net_results, key=lambda x: x['mean_mse'])['alpha']
elastic_net.set_params(alpha=best_en_alpha)
elastic_net.fit(X_train_scaled, y_train)
y_pred_en = elastic_net.predict(X_test_scaled)
en_mse = mean_squared_error(y_test, y_pred_en)
en_r2 = r2_score(y_test, y_pred_en)
print(f"\nBest Elastic Net Alpha: {best_en_alpha}")
print(f"Elastic Net Test MSE: {en_mse}, R2: {en_r2}")

# Feature Importance from Lasso
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(lasso.coef_)
}).sort_values(by='Importance', ascending=False)

print("Feature Importance from Lasso:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Visualization of Results
results = pd.DataFrame({
    'Model': ['Base Regression', 'Ridge', 'Lasso', 'Elastic Net'],
    'MSE': [base_mse, ridge_mse, lasso_mse, en_mse],
    'R-squared': [base_r2, ridge_r2, lasso_r2, en_r2]
})

print("\nComparison of Models:")
print(results)

# Plot comparison of MSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results)
plt.title('Comparison of MSE Across Models')
plt.ylabel('Mean Squared Error')
plt.show()



# Select top features based on importance threshold
importance_threshold = 0.01  
top_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature']

print(f"\nSelected Features based on importance (Threshold = {importance_threshold}):")
print(top_features.tolist())

# Filter the dataset to only include selected features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Train and evaluate a Ridge model on the selected features
ridge = Ridge(alpha=20, random_state=42)  
ridge.fit(X_train_selected, y_train)
y_pred_r = ridge.predict(X_test_selected)

# Evaluate the Ridge model
ridge_mse = mean_squared_error(y_test, y_pred_r)
ridge_r2 = r2_score(y_test, y_pred_r)

print(f"\nRidge Model Evaluation with Selected Features:")
print(f"Mean Squared Error: {ridge_mse}")
print(f"R-squared: {ridge_r2}")

# Train and evaluate a Lasso model on the selected features
lasso = Lasso(max_iter=10000, alpha = 0.1)
lasso.fit(X_train_selected, y_train)
y_pred_l = lasso.predict(X_test_selected)

# Evaluate the Lasso model
lasso_mse = mean_squared_error(y_test, y_pred_l)
lasso_r2 = r2_score(y_test, y_pred_l)

print(f"\nLasso Model Evaluation with Selected Features:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"R-squared: {lasso_r2}")


################################################################################################### 







