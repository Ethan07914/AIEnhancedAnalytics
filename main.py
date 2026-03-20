import requests
from config import api_key
import sys
import pandas as pd
import logging
import datetime as dt
from test import is_empty

def extract(url="https://newsapi.org/v2/top-headlines",
            api_key=api_key,
            date=dt.datetime.now(),
            language='en',
            sort_by='relevancy'):
    params = {
        'apiKey': api_key,
        'from_param': date,
        'to': date,
        'language': language,
        'sort_by': sort_by
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        message = f"API request successful, status: {data['status']}, result count: {data['totalResults']}"
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"API request failed: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return data

extracted = extract()

is_empty(extracted, dict)

# columns = ['source', 'author', 'title', 'description', 'url', 'urlToImage', 'publishedAt', 'content'],

def transform(data):
    try:
        df = pd.DataFrame(data['articles'])
        df['source_name'] = df.apply(lambda row: row['source']['name'], axis=1)
        df = df.drop(columns=['url', 'urlToImage', 'content', 'source'])
        file_path = 'transformed.csv'
        df.to_csv(file_path, index=False)
        message = f"Data transformed successfully and saved to file: {file_path}"
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"Data transformation failed: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return df


transformed = transform(extracted)

is_empty(transformed, pd.DataFrame)

print(transformed)