from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data # shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names,iris.target_names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("predictions:", y_pred[:5])
print("True labels:", y_test[:5])


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

import joblib

joblib.dump(model, "decision_tree_model.joblib")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm = confusion_matrix(y_test, y_pred)


print("Confusion Matrix:\n", cm)


plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.colorbar()


classes = iris.target_names
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)


for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


plt.savefig("confusion_matrix.png")


plt.show()

print("Confusion matrix image saved as 'confusion_matrix.png'")























