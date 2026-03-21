from transformers import pipeline
import pandas as pd
import ast
import logging
import sys
from test import file_exists
import os
from dotenv import load_dotenv

load_dotenv()

def topic_classifier(file_path,
                     labels=["world", "politics", "business", 'technology', 'science', 'health',
                             'entertainment', 'travel', 'food & drink', 'fashion', 'environment']
                     ):
    file_exists(file_path)
    try:
        descriptions = pd.read_csv(file_path)['description'].tolist()
        topic_classifier = pipeline("zero-shot-classification", model="tasksource/deberta-small-long-nli", token=os.getenv('hf_token'))
        topic_classifications = topic_classifier(descriptions, labels)
        df = pd.DataFrame(topic_classifications)
        try:
            df['scores'] = df['scores'].apply(ast.literal_eval)
            df['labels'] = df['labels'].apply(ast.literal_eval)
        except Exception as e:
            logging.error(e)
        df['label_probability'] = df.apply(lambda row: row['scores'][0], axis=1)
        df['label'] = df.apply(lambda row: row['labels'][0], axis=1)
        df = df.drop(columns=['scores', 'labels'])
        file_path = 'topic.csv'
        df['ID'] = range(len(df))
        df = df[['ID', 'label', 'label_probability']]
        df.to_csv(file_path, index=False)
        message = f"Successfully classified the topics of {len(df)} records into {file_path}."
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"Error occurred during topic classification: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return topic_classifications

topic_classifier('transformed.csv')

def sentiment_classifier(file_path):
    file_exists(file_path)
    try:
        descriptions = pd.read_csv(file_path)['description'].tolist()
        text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", token=os.getenv('hf_token'))
        text_classifications = text_classifier(descriptions)
        df = pd.DataFrame({'classification' : text_classifications, 'description': descriptions})
        df['sentiment'] = df.apply(lambda row: row['classification']['label'], axis=1)
        df['sentiment_probability'] = df.apply(lambda row: row['classification']['score'], axis=1)
        df = df.drop('classification', axis=1)
        file_path = 'sentiment.csv'
        df['ID'] = range(len(df))
        df = df[['ID', 'sentiment', 'sentiment_probability']]
        df.to_csv(file_path, index=False)
        message = f"Successfully classified the sentiment of {len(df)} records into {file_path}."
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"Error occurred during sentiment classification: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return text_classifications


sentiment_classifier('transformed.csv')



transformed_df = pd.read_csv('transformed.csv')
sentiment_df = pd.read_csv('sentiment.csv')
topic_df = pd.read_csv('topic.csv')

result = pd.merge(transformed_df, sentiment_df, how='inner', on='ID')
result = pd.merge(result, topic_df, how='inner', on='ID')

result['published_at'] = pd.to_datetime(result['publishedAt']).dt.date

result[['ID', 'source_name', 'title', 'author', 'published_at', 'description',
        'label' , 'sentiment', 'label_probability', 'sentiment_probability']].to_csv('article_fct.csv', index=False)
