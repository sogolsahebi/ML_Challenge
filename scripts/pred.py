import numpy as np
from model import softmax
from data_processing import clean_data, normalize_data, feature_choice
import pandas as pd
from model import grad_descent


def process_data(data):
    data = clean_data(data)
    data = normalize_data(data)
    X = feature_choice(data)
    x_train = X.drop(["Label"], axis=1).values
    labels = pd.get_dummies(X["Label"])
    t_train = labels.to_numpy()
    weights = grad_descent(X_train=x_train, t_train=t_train, alpha=0.1, n_iter=30)
    label_mapping = {i: label for i, label in enumerate(labels.columns)}
    return x_train, weights, label_mapping


def predict_all(filename: str):
    data = pd.read_csv(filename)
    X, weights, label_mapping = process_data(data)
    z = np.dot(X, weights)
    prob = softmax(z)
    predictions = np.argmax(prob, axis=1)
    prediction_labels = [label_mapping[pred] for pred in predictions]
    return prediction_labels
