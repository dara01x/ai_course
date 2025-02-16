from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
print('Iris dataset:')
print('X-size = ', X_iris.shape)
print('y-size = ', y_iris.shape)
print(X_iris[0:5, :])