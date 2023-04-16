from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Dictionary of activity levels
activity_levels = {
    "sedentary": 1.2,
    "lightly active": 1.375,
    "moderately active": 1.55,
    "very active": 1.725,
    "super active": 1.9
}

# Diet recommendation system code
def diet_recommendation(age, height, weight, gender, activity_level, dietary_restrictions, health_goals, food_dislikes):
    # Load food database
    food_data = pd.read_csv('D:/Coding/HTML/Website/Flask/food_database.csv')

    # Filter food database based on user preferences
    food_data = food_data[food_data['calories'] <= (weight * 25)] # Limit calories based on weight
    if dietary_restrictions != "":
        food_data = food_data[food_data[dietary_restrictions] == 0] # Filter out restricted foods
    if food_dislikes != "":
        food_data = food_data[~food_data['name'].str.contains(food_dislikes, case=False)] # Filter out disliked foods

    # Standardize nutritional data
    nutritional_data = food_data[['calories', 'protein', 'fat', 'carbs', 'fiber', 'vitamin_a', 'vitamin_c', 'calcium', 'iron']]
    scaler = StandardScaler()
    nutritional_data = scaler.fit_transform(nutritional_data)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    nutritional_data_pca = pca.fit_transform(nutritional_data)

    # Cluster foods using K-means
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(nutritional_data_pca)
    food_data['cluster'] = kmeans.labels_

    # Find nearest neighbors to user data
    user_data = np.array([weight, height, age, gender, activity_levels[activity_level]])
    user_data = np.append(user_data, np.zeros(4)) # Pad with zeros for nutritional data
    user_data = user_data.reshape(1, -1)

    # Create a LabelEncoder object to encode the "gender" value
    gender_encoder = LabelEncoder()

    # Encode the "gender" value as a numeric feature
    user_data[0][3] = gender_encoder.fit_transform([gender])[0]

    scaler = StandardScaler()
    user_data = scaler.fit_transform(user_data)
    user_data_pca = pca.transform(user_data)
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(nutritional_data_pca)
    distances, indices = knn.kneighbors(user_data_pca)
    recommended_cluster = kmeans.labels_[indices[0][0]]
    recommended_foods = food_data[food_data['cluster'] == recommended_cluster]['name'].tolist()

    return recommended_foods

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for recommendation page
@app.route('/recommendation.html', methods=['POST'])
def recommendation():
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    gender = request.form['gender']
    activity_level = request.form['activity_level']
    dietary_restrictions = request.form['dietary_restrictions']
    health_goals = request.form['health_goals']
    food_dislikes = request.form['food_dislikes']
    recommended_foods = diet_recommendation(age, height, weight, gender, activity_level, dietary_restrictions, health_goals, food_dislikes)
    return render_template('recommendation.html', recommended_foods=recommended_foods)


if __name__ == '__main__':
    app.run()

