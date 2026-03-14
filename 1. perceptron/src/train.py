from perceptron import Perceptron
from utils import generate_dataset, split_dataset, accuracy, plot_decision_boundary

X, y = generate_dataset()

X_train, X_test, y_train, y_test = split_dataset(X, y)

model = Perceptron(lr=0.01, epochs=50)
model.fit(X_train, y_train)

preds = [model.predict(x) for x in X_test]

print("Accuracy:", accuracy(y_test, preds))

plot_decision_boundary(model, X, y)