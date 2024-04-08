import numpy as np
import pandas as pd
from data_processing import clean_data, feature_choice, normalize_data
from model import softmax


def read_weights_from_file(filename):
    weights = []
    with open(filename, 'r') as file:
        for line in file:
            new_line = line.strip().replace('[', '').replace(']', '').replace(
                ',', ' ')
            row = [float(num) for num in new_line.split()]
            weights.append(row)
    return np.array(weights)


WEIGHTS = read_weights_from_file('weights.txt')


def process_data(data):
    data = clean_data(data)
    x = normalize_data(data)
    x_features = feature_choice(x)
    return x_features


def predict_all(filename: str):
    data = pd.read_csv(filename)
    x = process_data(data)
    z = np.dot(x, WEIGHTS)
    prob = softmax(z)
    predictions = np.argmax(prob, axis=1)

    # Map indices to city names
    cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
    return [cities[idx] for idx in predictions]
