import requests
import sys
import pandas as pd
import logging
import os
import datetime as dt
import json
from test import file_exists
from dotenv import load_dotenv
from prefect import task

load_dotenv()

@task
def extract(url="https://newsapi.org/v2/top-headlines",
            api_key=os.getenv('api_key'),
            date=dt.datetime.now(),
            language='en',
            sort_by='relevancy',
            page_size=100,
            page=1):
    params = {
        'apiKey': api_key,
        'from_param': date,
        'to': date,
        'language': language,
        'sort_by': sort_by,
        'pageSize': page_size,
        'page': page
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        with open('extracted.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
        message = f"API request successful, status: {data['status']}, result count: {data['totalResults']}"
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"  API request failed: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
@task
def transform():
    try:
        with open('extracted.json', 'r') as infile:
            data = json.load(infile)
        df = pd.DataFrame(data['articles'])
        df['source_name'] = df.apply(lambda row: row['source']['name'], axis=1)
        df = df.drop(columns=['url', 'urlToImage', 'content', 'source'])
        file_path = 'transformed.csv'
        df = df.fillna('Unknown')
        df = df.replace('', 'Unknown')
        df['ID'] = range(len(df))
        df = df[['ID', 'author', 'title' , 'description', 'publishedAt', 'source_name']]
        df.to_csv(file_path, index=False)
        message = f"Data transformed successfully and saved to file: {file_path}"
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"  Data transformation failed: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return df

@task
def load():
    file_exists('topic.csv') and file_exists('sentiment.csv') and file_exists('transformed.csv')
    try:
        transformed_df = pd.read_csv('transformed.csv')
        sentiment_df = pd.read_csv('sentiment.csv')
        topic_df = pd.read_csv('topic.csv')

        result = pd.merge(transformed_df, sentiment_df, how='inner', on='ID')
        result = pd.merge(result, topic_df, how='inner', on='ID')

        result['published_at'] = pd.to_datetime(result['publishedAt']).dt.date

        result = result[['ID', 'source_name', 'title', 'author', 'published_at', 'description',
                'label' , 'sentiment', 'label_probability', 'sentiment_probability']]
        result.to_csv('aienhancedanalytics/seeds/article.csv', index=False)
        message = f"Successfully enriched {len(result)} records."
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"Error occurred during enrichment: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()

