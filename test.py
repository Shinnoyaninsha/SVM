from sklearn.datasets import make_moons, make_classification
import seaborn as sns
import matplotlib.pyplot as plt

# Générer les données non linéaires
X, y = make_classification(n_samples=50, random_state=42)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from weighted_svm import WeightedSVM
model = WeightedSVM(kernel='poly', n_iters=100)
model.fit(X_train, y_train)
print(model.alpha)
