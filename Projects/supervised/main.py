from src.preprocess import load_and_preprocess
from src.train import train_model
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = load_and_preprocess('data/dataset.csv')


model = train_model(X_train, y_train)

y_pred = evaluate_model(model, X_test, y_test)


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title("Prediction Visualization")
plt.xlabel("Age (scaled)")
plt.ylabel("Estimated Salary (scaled)")
plt.show()