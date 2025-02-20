import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Calculate the variance captured by each principal component
explained_variance = pca.explained_variance_ratio_
total_variance = explained_variance.sum()

# Print the variance captured by each component and the total variance
print(f"Varianza explicada por cada componente: {explained_variance}")
print(f"Varianza total capturada: {total_variance}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')
joblib.dump(pca, 'pca_transform.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Precisión: {accuracy}")
print("Matriz de Confusión:")
print(conf_matrix)
print("Reporte de Clasificación:")
print(class_report)

# Create a DataFrame for plotting
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df['target'] = y
df['target_name'] = df['target'].apply(lambda i: iris.target_names[i])

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

# Add legend with accuracy and variance information
legend_text = (f"Precisión: {accuracy:.2f}\n"
            f"Varianza Capturada: {total_variance:.2f}\n"
            f"PC1: {explained_variance[0]:.2f}\n"
            f"PC2: {explained_variance[1]:.2f}\n"
            f"PC3: {explained_variance[2]:.2f}")
plt.legend(title=legend_text, loc='upper right')

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('Dataset IRis con PCA 3 componentes')

# Save the plot as a PNG file
plot_path = 'static/iris_pca_plot.png'
plt.savefig(plot_path)
plt.show()