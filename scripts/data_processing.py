import pandas as pd
from sklearn.feature_extraction.text import \
    TfidfVectorizer  # For converting text to numerical data
from sklearn.preprocessing import OneHotEncoder, \
    MinMaxScaler  # For encoding categorical variables and normalizing numerical data


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
            row[f'Q6_{category}'] = default_value
    else:
        for pair in entry.split(','):
            parts = pair.split('=>')
            if len(parts) == 2 and parts[1].isdigit():
                row[f'Q6_{parts[0]}'] = int(parts[1])
            else:
                row[f'Q6_{parts[0]}'] = default_value

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
        data[col].fillna(data[col].median(), inplace=True)

    categorical_cols = ['Q5', 'Q10']
    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)

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
    for column in ['Q6_Skyscrapers', 'Q6_Sport', 'Q6_Art and Music',
                   'Q6_Carnival', 'Q6_Cuisine', 'Q6_Economic']:
        data_clean[column] = data_clean[column].clip(
            lower=0)
    return data_clean


def normalize_data(data):
    """
    Normalizes Data
    :param data:
    :return:
    """
    processed_data = data.copy()
    scaler = MinMaxScaler()
    numerical_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q8', 'Q9',
                      'Q6_Skyscrapers', 'Q6_Sport', 'Q6_Art and Music',
                      'Q6_Carnival',
                      'Q6_Cuisine', 'Q6_Economic']
    processed_data[numerical_cols] = processed_data[numerical_cols].fillna(0)

    processed_data[numerical_cols] = scaler.fit_transform(
        processed_data[numerical_cols])

    one_hot_encoder = OneHotEncoder()
    q5_encoded = one_hot_encoder.fit_transform(
        processed_data[['Q5']]).toarray()  # Directly convert to array here

    feature_names = one_hot_encoder.get_feature_names_out(
        input_features=['Q5'])
    for i, column_name in enumerate(feature_names):
        processed_data[column_name] = q5_encoded[:, i]

    # Q10: Convert textual data in Q10 into numerical format using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    q10_tfidf = tfidf_vectorizer.fit_transform(processed_data['Q10']).toarray()

    # Get feature names from TF-IDF vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create DataFrame from TF-IDF vectors and concatenate with processed_data
    q10_tfidf_df = pd.DataFrame(q10_tfidf,
                                columns=['Q10_tfidf_' + name for name in
                                         feature_names])
    processed_data = pd.concat([processed_data.reset_index(drop=True),
                                q10_tfidf_df.reset_index(drop=True)], axis=1)

    processed_data.drop(['id', 'Q5', 'Q6', 'Q10'], axis=1, inplace=True)

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
        "Viral Potential",  # Q2
        "Avg Temp Jan",  # Q7
        "Q6_Economic",
        'Q6_Skyscrapers',
        "Label",
        "Fashion Styles"
        # "Sport",
        # 'Art and Music',
        # 'Carnival',
        # 'Economic'
    ]

    columns_all_keep = list(
        set(columns_keep + additional_columns_keep))  # columns_keep +

    data_keep = data[columns_all_keep]
    return data_keep
