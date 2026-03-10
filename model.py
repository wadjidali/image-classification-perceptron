from sklearn.linear_model import Perceptron

def train_model(X, y):
    
    model = Perceptron()
    model.fit(X, y)
    
    return model
