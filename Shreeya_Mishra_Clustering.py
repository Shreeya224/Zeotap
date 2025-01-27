import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

#Merge datasets
transaction_summary = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',  
    'Quantity': 'sum'    
}).reset_index()

customer_data = pd.merge(customers, transaction_summary, on='CustomerID', how='left').fillna(0)

#  Feature engineering

customer_data['SignupDate'] = pd.to_datetime(customer_data['SignupDate'])
customer_data['DaysSinceSignup'] = (pd.Timestamp.now() - customer_data['SignupDate']).dt.days
customer_data = customer_data.drop(columns=['SignupDate', 'CustomerName'])

customer_data = pd.get_dummies(customer_data, columns=['Region'], drop_first=True)

# Data normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(customer_data.drop(columns=['CustomerID']))

# K-Means with a range of clusters 
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    db_index = davies_bouldin_score(normalized_data, cluster_labels)
    db_scores.append((n_clusters, db_index))

#  number of clusters based on DB Index
best_clusters = min(db_scores, key=lambda x: x[1])[0]
print(f"Optimal number of clusters: {best_clusters}")

# clustering
kmeans = KMeans(n_clusters=best_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(normalized_data)

#Evaluation metrics
final_db_index = davies_bouldin_score(normalized_data, customer_data['Cluster'])
print(f"Davies-Bouldin Index for {best_clusters} clusters: {final_db_index}")

#  Visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)
customer_data['PCA1'] = reduced_data[:, 0]
customer_data['PCA2'] = reduced_data[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=customer_data, palette='tab10', s=100)
plt.title(f'Customer Clusters ({best_clusters} clusters)')
plt.show()

#  results and code
customer_data.to_csv('C:/Users/user/Desktop/Customer_Segmentation.csv', index=False)
print("Clustered customer data saved as 'Customer_Segmentation.csv'.")

script_path = 'C:/Users/user/Desktop/Customer_Clustering_Script.py'
with open(script_path, 'w') as script_file:
    script_file.write("""
# Clustering script for customer segmentation.
""")
print(f"Clustering script saved as 'Customer_Clustering_Script.py'.")
