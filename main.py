# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns  # New import for enhanced visualizations

# Importing data and exploratory data analysis to prepare data for the model
songs = pd.read_csv("/kaggle/input/worlds-spotify-top-50-playlist-musicality-data/Top-50-musicality-global.csv")
pop_scores = []

# Categorizing songs based on popularity scores
for i in songs["Popularity"]:
    if i > 80:
        pop_scores.append("Very popular")
    elif i > 60:
        pop_scores.append("Popular")
    elif i > 40:
        pop_scores.append("Middle popularity")
    else:
        pop_scores.append("Unpopular")

# Adding a new column to the dataframe to represent popularity ranks
songs["Popularity rank"] = pop_scores

# Displaying a subset of the dataframe for inspection
print(songs[["Energy", "Liveness", "Acousticness", "Popularity", "Popularity rank"]].head())

# Additional Analysis and Visualizations

# 1. Identify the most popular songs globally
global_top_songs = songs.groupby('Song')['Country'].count().sort_values(ascending=False).head(10)
print("Top Songs Globally:")
print(global_top_songs)

# 2. Analyze the distribution of musical aspects
plt.figure(figsize=(10, 6))
sns.histplot(data=songs[['Energy', 'Liveness', 'Acousticness']], kde=True)
plt.title('Distribution of Musical Aspects')
plt.xlabel('Musical Aspect Value')
plt.ylabel('Frequency')
plt.show()

# 3. Explore musical characteristics across countries
country_musical_trends = songs.groupby('Country')[['Energy', 'Liveness', 'Acousticness']].mean()
print("Country-wise Musical Characteristics:")
print(country_musical_trends)

# 4. Determine countries with the most diverse musical preferences
diversity_by_country = songs.groupby('Country')['Song'].nunique().sort_values(ascending=False)

# 5. Examine the distribution of publication dates
sns.histplot(data=songs, x='Publication Date', bins=20, kde=True)
plt.title('Distribution of Publication Dates')
plt.show()

# 6. Identify most frequently featured artists globally
top_artists_global = songs['Artist'].value_counts().head(10)

# 7. Investigate the correlation between musical aspects and popularity ranking
correlation_matrix = songs[['Energy', 'Liveness', 'Acousticness', 'Popularity']].corr()

# 8. Track the consistency of song popularity within a country over time
popularity_over_time = songs.groupby(['Country', 'Song'])['Popularity'].mean()

# 9. Identify songs with regional popularity variations
regional_variations = songs.groupby(['Song', 'Country']).size().unstack(fill_value=0)

# 10. Conduct outlier analysis for song popularity
sns.boxplot(x='Popularity rank', y='Popularity', data=songs)
plt.title('Boxplot of Song Popularity by Rank')
plt.xlabel('Popularity Rank')
plt.ylabel('Popularity')
plt.show()

# Making model, as well as setting a for loop to iterate through different neighbor values
X = songs[["Energy", "Liveness", "Acousticness"]]
y = songs["Popularity rank"]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y)

n_range = np.arange(3, 20)

train_accuracies = []
test_accuracies = []

# Looping through different neighbor values and fitting the KNN model
for z in n_range:
    knn = KNeighborsClassifier(n_neighbors=z)
    knn.fit(X_train, y_train)
    
    # Recording the training and testing accuracies
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

# Making a graph, to show the training and testing accuracy for different neighbor values
fig, ax = plt.subplots()
plt.plot(n_range, train_accuracies, label="Training accuracies")
plt.plot(n_range, test_accuracies, label="Testing accuracies")
plt.legend()
plt.title("KNN classifier with different number of neighbors")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")

# Displaying the graph
plt.show()
