# ml_model_00
Implements a model with the Iris Dataset using flask and a simple API to connect to it and make predictions

It could be any model. The idea, is when you pull the model to your server, run train_model.py to generate the necessary .pkl file (train it on the server).
Then run app.py to start the application.

Then the api works with:
API Endpoints
1. /login (POST)
Description: Authenticates the user and returns a JWT token.

Request:
{
  "username": "admin",
  "password": "password"
}
Response:
{
  "access_token": "your_jwt_token"
}

2. /predict (POST)
Description: Predicts the variety of Iris flower based on the input features.

Request:
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
Response:
{
  "prediction": "setosa",
  "input_data": {
    "Sepal Length": 5.1,
    "Sepal Width": 3.5,
    "Petal Length": 1.4,
    "Petal Width": 0.2
  }
}

3. /auto_predict (GET)
Description: Randomly selects a row from the dataset for prediction and returns the result.

Response:
{
  "prediction": "setosa",
  "input_data": {
    "Sepal Length": 5.1,
    "Sepal Width": 3.5,
    "Petal Length": 1.4,
    "Petal Width": 0.2
  }
}

4. /plot (GET)
Description: Returns the plot with the predicted point highlighted.

Response: Returns an image file (iris_pca_plot_with_prediction.png).

Usage Examples
Login:
curl -X POST http://localhost:5000/login -H "Content-Type: application/json" -d '{"username":"admin","password":"password"}'
Predict:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -H "Authorization: Bearer your_jwt_token" -d '{"features":[5.1, 3.5, 1.4, 0.2]}'
Auto Predict:
curl -X GET http://localhost:5000/auto_predict
Get Plot:
curl -X GET http://localhost:5000/plot --output iris_pca_plot_with_prediction.png



