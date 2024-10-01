#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:06:46 2024

@author: Tejinder
@student_id: 301232634
"""

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

#Step1 - I will load the Olivetti faces dataset from sklearn datasets
data = fetch_olivetti_faces(shuffle=True, random_state=34)
#Extract X and y
X = data.data
y = data.target
#Step 2 - Stratified split ensuring same number of images per person in all splits

#Ratio for the split, I chose 60, 20, 20
#I didnot choose 80, 10, 10 because that would mean only one image per person for test and validation
#We can not choose something like 70, 15, 15 because then we can not have equal iamges
#60, 20, 20 means 6, 2, 2 images per person in the split
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

#Stratified split for training data and temp data (data containing val and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(1 - train_ratio), stratify=y, random_state=42)

#Stratified split for val and test data from the temp data
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(test_ratio / (test_ratio + validation_ratio)), 
    stratify=y_temp, random_state=42)

#Method to count the number of images per person
def count_images_per_person(y_data, dataset_name):
    unique, counts = np.unique(y_data, return_counts=True)
    return pd.DataFrame({'Person': unique, dataset_name: counts})

#Lets confirm the number
train_counts = count_images_per_person(y_train, 'Training')
validation_counts = count_images_per_person(y_val, 'Validation')
test_counts = count_images_per_person(y_test, 'Test')
train_counts
validation_counts
test_counts

#Merging all the counts to be plot togeather
merged_counts = train_counts.merge(validation_counts, on='Person').merge(test_counts, on='Person')

#Visualizing the split
plt.figure(figsize=(12, 6))
merged_counts.plot(kind='bar', x='Person', stacked=False, figsize=(12,6))
plt.title('Number of Images Per Person in Training, Validation, and Test Sets')
plt.xlabel('Person ID')
plt.ylabel('Number of Images')
plt.grid(True)
plt.show()

#Step 3 - I am going to use svc linear classifer to make predictions on val dataset
clf = SVC(kernel='linear', random_state=42)

#Stratified K-Fold Cross Validation with 5 splits
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Cross-validation on the training set
cross_val_scores = cross_val_score(clf, X_train, y_train, cv=kf)

#Training the classifier
clf.fit(X_train, y_train)

#Prediction on the validation set
y_val_pred = clf.predict(X_val)

#Calculating the accuracy on the validation set
validation_accuracy = accuracy_score(y_val, y_val_pred)

#Output the results
print(f'Cross-Validation Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Accuracy: {cross_val_scores.mean()}')
print(f'Validation Set Accuracy: {validation_accuracy}')

#Step 4 - K-means and Silhoutte score
#Visualizing the original faces

#Get unique person indices using the target class (0 to 39)
unique_people_indices = np.unique(data.target, return_index=True)[1]

#There are 40 individuals
n_people = 40

#Prepare to plot one image for each unique person
fig, axes = plt.subplots(4, 10, figsize=(15, 6), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.5, wspace=0.3))

#Loop to plot one unique image for each person based on the target class
for i, ax in enumerate(axes.flat):
    person_index = unique_people_indices[i]
    ax.imshow(data.images[person_index], cmap='gray')
    ax.set_title(f'Person {data.target[person_index]+1}')

plt.suptitle("Olivetti Faces - One Unique Image Per Person", size=16)
plt.show()

#There are 40 different people in the images so I must set the rang from 2 to 50
range_n_clusters = list(range(2, 50))
inertia_values = []
silhouette_avg_scores = []

#Perform KMeans for different numbers of clusters
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=34, n_init=10) #warning on using auto
    cluster_labels = kmeans.fit_predict(X_train)
    
    inertia_values.append(kmeans.inertia_) #inertia
    #silhouette score
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

#Plotting elbow method to look for the elbow
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertia_values, marker='o')
plt.title('Elbow Method: Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.grid(True)
plt.show()

#Plotting silhouette scores for different numbers of clusters
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
plt.title('Silhouette Scores vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

#Get the top 10 silhouette scores along with their number of clusters
top_10_indices = sorted(range(len(silhouette_avg_scores)), key=lambda i: silhouette_avg_scores[i], reverse=True)[:10]
top_10_clusters_and_scores = [(range_n_clusters[i], silhouette_avg_scores[i]) for i in top_10_indices]

#Print the top 10 silhouette scores and their corresponding number of clusters
print("Top 10 Silhouette Scores and Corresponding Number of Clusters:")
for n_clusters, score in top_10_clusters_and_scores:
    print(f"Number of Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

#Applying KMeans with optimal number of clusters
optimal_clusters = 40
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
X_train_kmeans = kmeans.fit_transform(X_train)

#Dimensionality Reduction using cluster affinities
#We replace each instance's original feature vector with its affinity vector (distance to each cluster center)
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)
X_test_reduced = kmeans.transform(X_test)

#Now X_train_reduced, X_val_reduced, and X_test_reduced have 40 dimensions
print(f"Original training set dimensions: {X_train.shape}")
print(f"Reduced training set dimensions (after K-Means): {X_train_reduced.shape}")

#Training classifier on trained data
from sklearn.svm import SVC
clf = SVC(kernel='linear', random_state=42)

#Training the classifier on the reduced training set
clf.fit(X_train_reduced, y_train)

#Evaluating the classifier on the reduced validation set
y_val_pred = clf.predict(X_val_reduced)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation accuracy after dimensionality reduction using KMeans: {val_accuracy}')

#Step 6 - DBSCAN
#Let's standardize the data as form of preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Function to apply DBSCAN with varying eps and find the best number of clusters
def find_best_eps(X, eps_start, eps_end, step, min_samples=2):
    best_eps = None
    closest_n_clusters = None
    best_noise_ratio = None

    for eps in np.arange(eps_start, eps_end, step):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)

        #Calculating the number of clusters and noise points
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = list(dbscan.labels_).count(-1)
        noise_ratio = n_noise / len(X)

        print(f"DBSCAN with eps={eps:.2f}: {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%} noise)")

        #I want to find a reasonable number of clusters and minimize noise
        if 30 <= n_clusters <= 50:
            if best_noise_ratio is None or noise_ratio < best_noise_ratio:
                best_eps = eps
                closest_n_clusters = n_clusters
                best_noise_ratio = noise_ratio

    return best_eps, closest_n_clusters, best_noise_ratio

best_eps, n_clusters, noise_ratio = find_best_eps(X_scaled, eps_start=25, eps_end=65.0, step=1)

if best_eps is not None:
    print(f"Best eps value: {best_eps}, forming {n_clusters} clusters with {noise_ratio:.2%} noise")
else:
    print("No suitable eps value was found in the given range.")

#Applying DBSCAN with the best eps value and plotting the results
def apply_dbscan_and_plot(X, eps):
    dbscan = DBSCAN(eps=eps, min_samples=2)
    dbscan.fit(X)

    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = list(dbscan.labels_).count(-1)
    
    print(f"DBSCAN with eps={eps}")
    print(f"Number of clusters: {n_clusters}, Number of noise points: {n_noise}")

    if len(set(dbscan.labels_)) > 1:
        silhouette_avg = silhouette_score(X_scaled, dbscan.labels_)
        print(f"Silhouette Score for best DBSCAN: {silhouette_avg:.4f}")
    else:
        print("Only one cluster or all noise, silhouette score is not defined.")
    
    #Plotting the results
    plot_dbscan(dbscan, X, size=10)
    return dbscan

#Plot DBSCAN results for the best eps value
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)
    
    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20,
                c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".")
    
    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)
    
    plt.title(f"DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})")
    plt.grid()
    plt.gca().set_axisbelow(True)

#Applying DBSCAN and store the dbscan object
if best_eps is not None:
    dbscan = apply_dbscan_and_plot(X_scaled, best_eps)
else:
    print("Could not find an optimal eps value.")

#Defining function to plot cluster images
def plot_cluster_images(cluster_labels, X_images, n_clusters):
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        #Plotting all images in particular cluster
        print(f"\nCluster {cluster + 1} - {len(cluster_indices)} images")
        fig, axes = plt.subplots(1, len(cluster_indices), figsize=(15, 5))
        if len(cluster_indices) > 1:
            for idx, ax in zip(cluster_indices, axes):
                ax.imshow(X_images[idx], cmap='gray')
                ax.axis('off')
        else:
            axes.imshow(X_images[cluster_indices[0]], cmap='gray')
            axes.axis('off')
        
        plt.suptitle(f"Cluster {cluster + 1}")
        plt.show()

#Getting the number of clusters formed (excluding noise points)
n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)

#Visualizing images in each cluster
plot_cluster_images(dbscan.labels_, data.images, n_clusters)