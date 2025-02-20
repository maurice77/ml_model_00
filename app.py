from flask import Flask, request, jsonify, send_file, render_template
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from datetime import timedelta
import os
import random

API_USERS = {
    'maurice':'hola',
    'user':'pass',
}

app = Flask(__name__)
CORS(app)  # Enable CORS

app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Change this to a random secret key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=60)  # Set token expiration time
jwt = JWTManager(app)

# Load the model and PCA transformation
model = joblib.load('iris_model.pkl')
pca = joblib.load('pca_transform.pkl')

# Load the dataset for plotting
iris = load_iris()
X, y = iris.data, iris.target
X_pca = pca.transform(X)

# Create a DataFrame for plotting
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df['target'] = y
df['target_name'] = df['target'].apply(lambda i: iris.target_names[i])

def generate_plot(features, features_pca, predicted_class):
    # Plotting the classes in 3D using Matplotlib and Seaborn
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assign colors to each class
    colors = sns.color_palette("hsv", len(iris.target_names))
    for i, target_name in enumerate(iris.target_names):
        ax.scatter(
            df[df['target_name'] == target_name]['PC1'],
            df[df['target_name'] == target_name]['PC2'],
            df[df['target_name'] == target_name]['PC3'],
            label=target_name,
            color=colors[i],
            s=100
        )

    # Highlight the predicted point
    print(f"Plotting predicted point: {features_pca[0]}")  # Debug print
    ax.scatter(
        features_pca[0, 0], features_pca[0, 1], features_pca[0, 2],
        color='red', s=300, label='Predicted Point', edgecolor='w'
    )

    # Add text annotation for the predicted variety
    text_x = max(df['PC1']) - 2
    text_y = min(df['PC2']) + 1
    text_z = min(df['PC3']) - 0.2
    ax.text(text_x, text_y, text_z, f'{predicted_class}', 
    color='red', fontsize=12, weight='bold')

    # Draw a red line from the text to the predicted point
    ax.plot([text_x-0.2, features_pca[0, 0]], [text_y, features_pca[0, 1]], [text_z+0.05, features_pca[0, 2]], color='red')

    # Add legend with accuracy and variance information
    accuracy = model.score(X_pca, y)
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    legend_text = (f"Accuracy: {accuracy:.2f}\n"
                   f"Variance captured: {total_variance:.2f}\n"
                   f"PC1: {explained_variance[0]:.2f}\n"
                   f"PC2: {explained_variance[1]:.2f}\n"
                   f"PC3: {explained_variance[2]:.2f}")
    plt.legend(title=legend_text, loc='upper right')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA of Iris Dataset')

    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the plot as a PNG file
    plot_path = 'static/iris_pca_plot_with_prediction.png'
    plt.savefig(plot_path)
    plt.close()

@app.route('/get_prediction')
def get_prediction():
    return render_template('get_prediction.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    st_error = False
    if username not in API_USERS:
        st_error = True
    else:
        if API_USERS[username] != password:
            st_error = True

    if st_error:    
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    print(f"Features: {features}")  # Debug print
    features_pca = pca.transform(features)
    print(f"Features PCA: {features_pca}")  # Debug print
    prediction = model.predict(features_pca)
    predicted_class = iris.target_names[prediction[0]]

    generate_plot(features, features_pca, predicted_class)

    input_data = {
        "Sepal Length": features[0, 0],
        "Sepal Width": features[0, 1],
        "Petal Length": features[0, 2],
        "Petal Width": features[0, 3]
    }

    return jsonify({'input_data': input_data, 'prediction': predicted_class})

@app.route('/plot', methods=['GET'])
def get_plot():
    plot_path = 'static/iris_pca_plot_with_prediction.png'
    return send_file(plot_path, mimetype='image/png')

@app.route('/auto_predict', methods=['GET'])
def auto_predict():
    # Use a sample input for prediction
    random_index = random.randint(0, len(X)-1)
    sample_input = X[random_index] #[5.1, 3.5, 1.4, 0.2]  # Example input from the Iris dataset
    features = np.array(sample_input).reshape(1, -1)
    features_pca = pca.transform(features)
    prediction = model.predict(features_pca)
    predicted_class = iris.target_names[prediction[0]]

    generate_plot(features, features_pca, predicted_class)

    input_data = {
        "Sepal Length": features[0, 0],
        "Sepal Width": features[0, 1],
        "Petal Length": features[0, 2],
        "Petal Width": features[0, 3]
    }

    print('hola')
    print(jsonify({'input_data': input_data, 'prediction': predicted_class}))
    print('chao')
    return jsonify({'input_data': input_data, 'prediction': predicted_class})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)