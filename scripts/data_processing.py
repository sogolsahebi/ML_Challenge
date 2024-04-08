import math

import pandas as pd


def to_numeric(s):
    """Converts string `s` to a float.
    Invalid strings and NaN values will be converted to float('nan').
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


def parse_and_update_q6(row):
    entry = row['Q6']
    categories = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival',
                  'Cuisine', 'Economic']
    default_value = -1  # Placeholder for non-integer or missing values

    if pd.isnull(entry) or entry == "":
        for category in categories:
            row[f'{category}'] = default_value
    else:
        for pair in entry.split(','):
            parts = pair.split('=>')
            if len(parts) == 2 and parts[1].isdigit():
                row[f'{parts[0]}'] = int(parts[1])
            else:
                row[f'{parts[0]}'] = default_value

    return row


def clean_data(data):
    """
    Cleans Data
    :param data:
    :return:
    """
    data['Q7'] = data['Q7'].apply(to_numeric)
    data['Q9'] = data['Q9'].apply(to_numeric)

    numerical_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q8', 'Q9']
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())

    categorical_cols = ['Q5', 'Q10']
    for col in categorical_cols:
        data[col] = data[col].fillna('Unknown')

    data = data.apply(parse_and_update_q6, axis=1)
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)

    data_clean = data.copy()
    data_clean['Q9'] = pd.to_numeric(data_clean['Q9'], errors='coerce')
    data_clean['Q7'] = pd.to_numeric(
        data_clean['Q7'], errors='coerce').astype('float64')

    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
        data_clean[col] = data_clean[col].clip(lower=1, upper=5)
    data_clean['Q7'] = data_clean['Q7'].clip(lower=0, upper=55)
    data_clean['Q8'] = data_clean['Q8'].clip(lower=1, upper=9.5)
    data_clean['Q9'] = data_clean['Q9'].clip(lower=0, upper=20.5)
    for column in ['Skyscrapers', 'Sport', 'Art and Music',
                   'Carnival', 'Cuisine', 'Economic']:
        data_clean[column] = data_clean[column].clip(
            lower=0)
    return data_clean


def one_hot_transform(cat, cols, prefix):
    """
    Transform data into a one-hot vector, returns it
    :return:
    """
    one_hot_enc = {f"{prefix}_{category}": [] for category in cat}
    for entry in cols:
        for category in cat:
            one_hot_enc[f"{prefix}_{category}"].append(int(entry == category))
    return pd.DataFrame(one_hot_enc)


def minmax_scaler_transform(data):
    """
    Transform data matrix by min max scaling
    :return:
    """
    scale_data = data.copy()
    for col in scale_data.columns:
        p = scale_data[col]
        max_val = p.max()
        min_val = p.min()
        scale_data[col] = (p - min_val) / (max_val - min_val)
    return scale_data


def compute_freq(string: str):
    """
    Compute term frequency
    :return:
    """
    tf_dict = {}
    words = string.split()
    for word in words:
        if word in tf_dict:
            tf_dict[word] += 1
        else:
            tf_dict[word] = 1

    all_terms = len(words)
    for word in tf_dict:
        tf_dict[word] /= all_terms
    return tf_dict


def compute_inverse_text_freq(text_list):
    idf_dict = {}
    n = len(text_list)

    for text in text_list:
        for word in set(text.split()):
            if word in idf_dict:
                idf_dict[word] += 1
            else:
                idf_dict[word] = 1

    for word in idf_dict:
        idf_dict[word] = math.log(n + 1 / idf_dict[word] + 1) + 1

    return idf_dict


def tfidf_vectorizer(all_texts, max_features=150):
    """
    Uses TFID to vectorize words
    :return:
    """
    tfidf_list = []
    idfs = compute_inverse_text_freq(all_texts)

    for text in all_texts:
        tf = compute_freq(text)
        tfidf_dict = {}
        for word, t_value in tf.items():
            tfidf_dict[word] = t_value * idfs.get(word, 0)
        tfidf_list.append(tfidf_dict)

    all_tfidf = {}
    for txt_tfidf in tfidf_list:
        for word, score in txt_tfidf.items():
            if word in all_tfidf:
                all_tfidf[word] += score
            else:
                all_tfidf[word] = score

    top_features = sorted(all_tfidf, key=all_tfidf.get, reverse=True)[
                   :max_features]
    filtered_tfidfs = []
    for tfidfs in tfidf_list:
        filtered_tfidf = {word: score for word, score in tfidfs.items() if
                          word in top_features}
        filtered_tfidfs.append(filtered_tfidf)
    return filtered_tfidfs


def normalize_data(data):
    """
    Normalizes Data
    :param data:
    :return:
    """
    processed_data = data.copy()
    numerical_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q8', 'Q9',
                      'Skyscrapers', 'Sport', 'Art and Music',
                      'Carnival',
                      'Cuisine', 'Economic']
    processed_data[numerical_cols] = processed_data[numerical_cols].fillna(0)

    processed_data[numerical_cols] = minmax_scaler_transform(
        processed_data[numerical_cols])

    categories = processed_data['Q5'].unique()
    q5_encoded_df = one_hot_transform(categories, processed_data['Q5'],
                                      prefix="Q5")
    processed_data = pd.concat([processed_data, q5_encoded_df], axis=1)

    # Q10: Convert textual data in Q10 into numerical format using TF-IDF
    q10_words = processed_data['Q10'].fillna("").tolist()
    q10_tfidf = tfidf_vectorizer(q10_words[:150])

    # Get feature names from TF-IDF vectorizer
    q10_tfidf_df = pd.DataFrame.from_records(q10_tfidf).fillna(0)
    q10_tfidf_df.columns = ['Q10_tfidf_' + str(col) for col in
                            q10_tfidf_df.columns]

    # Create DataFrame from TF-IDF vectors and concatenate with processed_data
    processed_data = pd.concat([processed_data.reset_index(drop=True),
                                q10_tfidf_df.reset_index(drop=True)],
                               axis=1)

    processed_data.drop(['id', 'Q5', 'Q6', 'Q10'], axis=1, inplace=True)
    processed_data.fillna(0, inplace=True)

    # Renaming columns for clarity
    processed_data.rename(columns={
        'Q1': "Popularity",
        'Q2': "Viral Potential",
        'Q3': "Architectural Uniqueness",
        'Q4': "Party Enthusiasm",
        'Q7': "Avg Temp Jan",
        'Q8': "Languages Overheard",
        'Q9': "Fashion Styles",
    }, inplace=True)

    return processed_data


def feature_choice(data):
    """
    Functions that chooses features
    :param data:
    :return:
    """
    columns_keep = [
        column for column in data.columns if "Q10" in column
    ]
    additional_columns_keep = [
        "Viral Potential",
        "Avg Temp Jan",
        "Economic",
        'Skyscrapers',
        # "Label",
        "Q5_Partner",
        "Q5_Partner,Friends",
        "Q5_Friends",
        "Sport",
        'Art and Music',
        'Carnival',
        "Architectural Uniqueness",
        "Languages Overheard"
    ]

    columns_all_keep = list(
        set(columns_keep + additional_columns_keep))  # columns_keep +

    data_keep = data[columns_all_keep]
    return data_keep
