from transformers import pipeline
import pandas as pd
import ast
import logging
import sys
from test import file_exists
from config import hf_token

# This will download the model (only 268MB) and run it on your CPU
def topic_classifier(file_path,
                     labels=["world", "politics", "business", 'technology', 'science', 'health',
                             'entertainment', 'travel', 'food & drink', 'fashion', 'environment']
                     ):
    file_exists(file_path)
    try:
        descriptions = pd.read_csv(file_path)['description'].tolist()
        topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=hf_token)
        topic_classifications = topic_classifier(descriptions, labels)
        df = pd.DataFrame(topic_classifications)
        try:
            df['scores'] = df['scores'].apply(ast.literal_eval)
            df['labels'] = df['labels'].apply(ast.literal_eval)
        except Exception as e:
            logging.error(e)
        df['scores'] = df.apply(lambda row: row['scores'][0], axis=1)
        df['labels'] = df.apply(lambda row: row['labels'][0], axis=1)
        df.to_csv('topics.csv')
        message = f"Successfully classified the topics of {len(df)} records."
        logging.info(message)
        print(message)
    except Exception as e:
        error_message = f"Error occurred during topic classification: {e}"
        logging.error(error_message)
        print(error_message)
        sys.exit()
    return topic_classifications

topic_classifier('data.csv')

