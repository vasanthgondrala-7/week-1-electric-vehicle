import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('ElectricCarData_Clean.csv')

print("=== ELECTRIC CAR DATASET GENAI ANALYSIS ===\n")

# 1. BASIC DATA EXPLORATION
print("1. DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"Number of brands: {df['Brand'].nunique()}")
print(f"Number of models: {df['Model'].nunique()}")
print("\nFirst few rows:")
print(df.head())

print("\n" + "="*50)
print("2. DATA QUALITY CHECK")
print("="*50)
print("Missing values:")
print(df.isnull().sum())
print(f"\nData types:\n{df.dtypes}")

# 2. DESCRIPTIVE STATISTICS
print("\n" + "="*50)
print("3. KEY STATISTICS")
print("="*50)
numeric_cols = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Efficiency_WhKm', 
               'FastCharge_KmH', 'Seats', 'PriceEuro']
print(df[numeric_cols].describe())

# 3. BRAND ANALYSIS
print("\n" + "="*50)
print("4. BRAND ANALYSIS")
print("="*50)
brand_stats = df.groupby('Brand').agg({
    'PriceEuro': ['count', 'mean', 'max'],
    'Range_Km': 'mean',
    'AccelSec': 'mean'
}).round(2)

brand_stats.columns = ['Model_Count', 'Avg_Price', 'Max_Price', 'Avg_Range', 'Avg_Acceleration']
brand_stats = brand_stats.sort_values('Avg_Price', ascending=False)
print("Top 10 brands by average price:")
print(brand_stats.head(10))

# 4. VISUALIZATION SECTION
print("\n" + "="*50)
print("5. DATA VISUALIZATIONS")
print("="*50)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Electric Car Dataset Analysis', fontsize=16, fontweight='bold')

# Plot 1: Price distribution by brand (top 15)
top_brands = df['Brand'].value_counts().head(15).index
price_data = df[df['Brand'].isin(top_brands)]
sns.boxplot(data=price_data, x='Brand', y='PriceEuro', ax=axes[0,0])
axes[0,0].set_title('Price Distribution by Brand (Top 15)')
axes[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Range vs Price colored by PowerTrain
sns.scatterplot(data=df, x='Range_Km', y='PriceEuro', hue='PowerTrain', ax=axes[0,1])
axes[0,1].set_title('Range vs Price by Power Train')

# Plot 3: Acceleration distribution
sns.histplot(df['AccelSec'], bins=20, kde=True, ax=axes[0,2])
axes[0,2].set_title('Acceleration Distribution (0-100 km/h)')
axes[0,2].set_xlabel('Acceleration (seconds)')

# Plot 4: Body style distribution
body_style_counts = df['BodyStyle'].value_counts()
axes[1,0].pie(body_style_counts.values, labels=body_style_counts.index, autopct='%1.1f%%')
axes[1,0].set_title('Body Style Distribution')

# Plot 5: Efficiency vs Fast Charge capability
sns.scatterplot(data=df, x='Efficiency_WhKm', y='FastCharge_KmH', 
                hue='RapidCharge', size='PriceEuro', ax=axes[1,1])
axes[1,1].set_title('Efficiency vs Fast Charge Capability')

# Plot 6: Segment analysis
segment_price = df.groupby('Segment')['PriceEuro'].mean().sort_values(ascending=False)
sns.barplot(x=segment_price.index, y=segment_price.values, ax=axes[1,2])
axes[1,2].set_title('Average Price by Vehicle Segment')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. CORRELATION ANALYSIS
print("\n" + "="*50)
print("6. CORRELATION ANALYSIS")
print("="*50)

# Calculate correlations
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# 6. CLUSTERING ANALYSIS
print("\n" + "="*50)
print("7. VEHICLE CLUSTERING ANALYSIS")
print("="*50)

# Prepare data for clustering
cluster_features = ['PriceEuro', 'Range_Km', 'AccelSec', 'TopSpeed_KmH']
cluster_df = df[cluster_features].dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters to dataframe
cluster_df = cluster_df.copy()
cluster_df['Cluster'] = clusters

# Analyze clusters
cluster_analysis = cluster_df.groupby('Cluster').mean()
print("Cluster Analysis (Average Values):")
print(cluster_analysis)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(cluster_df['Range_Km'], cluster_df['PriceEuro'], 
                     c=cluster_df['Cluster'], cmap='viridis', 
                     s=cluster_df['AccelSec']*10, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Range (Km)')
plt.ylabel('Price (Euro)')
plt.title('Vehicle Clusters: Range vs Price (Bubble size = Acceleration)')
for i, txt in enumerate(df.loc[cluster_df.index, 'Brand']):
    plt.annotate(txt, (cluster_df['Range_Km'].iloc[i], cluster_df['PriceEuro'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
plt.tight_layout()
plt.show()

# 7. PRICE PREDICTION MODEL
print("\n" + "="*50)
print("8. PRICE PREDICTION MODEL")
print("="*50)

# Prepare features for prediction
feature_columns = ['Range_Km', 'AccelSec', 'TopSpeed_KmH', 'Efficiency_WhKm', 
                  'FastCharge_KmH', 'Seats']

# Handle missing values in FastCharge_KmH
X = df[feature_columns].fillna(df[feature_columns].mean())
y = df['PriceEuro']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance for Price Prediction:")
print(feature_importance)

# 8. INSIGHTS AND RECOMMENDATIONS
print("\n" + "="*50)
print("9. KEY INSIGHTS AND RECOMMENDATIONS")
print("="*50)

# Generate insights
avg_price = df['PriceEuro'].mean()
avg_range = df['Range_Km'].mean()
avg_accel = df['AccelSec'].mean()

print(f"Market Overview:")
print(f"• Average electric car price: €{avg_price:,.0f}")
print(f"• Average range: {avg_range:.0f} km")
print(f"• Average acceleration (0-100 km/h): {avg_accel:.1f} seconds")

print(f"\nPremium Segment Analysis:")
premium_brands = ['Porsche', 'Lucid', 'Tesla', 'Audi', 'Mercedes']
premium_cars = df[df['Brand'].isin(premium_brands)]
print(f"• Number of premium models: {len(premium_cars)}")
print(f"• Average premium car price: €{premium_cars['PriceEuro'].mean():,.0f}")
print(f"• Average premium car range: {premium_cars['Range_Km'].mean():.0f} km")

print(f"\nBudget Segment Analysis:")
budget_cars = df[df['PriceEuro'] < 35000]
print(f"• Number of budget models (<€35K): {len(budget_cars)}")
print(f"• Average budget car range: {budget_cars['Range_Km'].mean():.0f} km")

print(f"\nEfficiency Leaders:")
efficient_cars = df.nsmallest(5, 'Efficiency_WhKm')[['Brand', 'Model', 'Efficiency_WhKm', 'Range_Km']]
print("Most efficient cars:")
for _, car in efficient_cars.iterrows():
    print(f"• {car['Brand']} {car['Model']}: {car['Efficiency_WhKm']} Wh/Km, Range: {car['Range_Km']} km")

print(f"\nPerformance Leaders:")
fast_cars = df.nsmallest(5, 'AccelSec')[['Brand', 'Model', 'AccelSec', 'TopSpeed_KmH', 'PriceEuro']]
print("Fastest accelerating cars:")
for _, car in fast_cars.iterrows():
    print(f"• {car['Brand']} {car['Model']}: {car['AccelSec']}s, Top Speed: {car['TopSpeed_KmH']} km/h, Price: €{car['PriceEuro']:,.0f}")

# 9. SEGMENT-BASED ANALYSIS
print("\n" + "="*50)
print("10. SEGMENT-BASED ANALYSIS")
print("="*50)

segment_analysis = df.groupby('Segment').agg({
    'PriceEuro': ['mean', 'count'],
    'Range_Km': 'mean',
    'AccelSec': 'mean',
    'Efficiency_WhKm': 'mean'
}).round(2)

segment_analysis.columns = ['Avg_Price', 'Model_Count', 'Avg_Range', 'Avg_Acceleration', 'Avg_Efficiency']
print(segment_analysis.sort_values('Avg_Price', ascending=False))

# 10. RAPID CHARGE ANALYSIS
print("\n" + "="*50)
print("11. RAPID CHARGE CAPABILITY ANALYSIS")
print("="*50)

rapid_charge_stats = df.groupby('RapidCharge').agg({
    'PriceEuro': 'mean',
    'FastCharge_KmH': 'mean',
    'Brand': 'count'
}).round(2)

print("Rapid Charge Capability Impact:")
print(rapid_charge_stats)

# Additional visualization: Price distribution by segment with rapid charge
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Segment', y='PriceEuro', hue='RapidCharge')
plt.title('Price Distribution by Segment and Rapid Charge Capability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)