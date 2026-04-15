from sklearn.linear_model import Ridge

def train_model(X, y):
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model