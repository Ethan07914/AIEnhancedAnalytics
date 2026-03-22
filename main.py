from nlp import sentiment_classifier, topic_classifier
from test import no_records_lost
from etl import extract, transform, load
from prefect import flow, task
import subprocess
import sys
import os
import logging

@task
def run_dbt_build(project_dir="aienhancedanalytics"):
    assert os.path.exists(project_dir)
    try:
        command = [
            "dbt", "build",
            "--project-dir", project_dir,
            "--profiles-dir", project_dir
        ]
        result = subprocess.run(command, check=True, text=True)
        message = "dbt build completed successfully!"
        logging.info(message)
        print(message)
    except subprocess.CalledProcessError as e:
        error_message = f"dbt build failed with return code {e.returncode}"
        logging.error(error_message)
        sys.exit()

@flow
def main():
    extract()
    transform()
    sentiment_classifier()
    no_records_lost('transformed.csv', 'sentiment.csv')
    topic_classifier()
    no_records_lost('transformed.csv', 'topic.csv')
    load()
    no_records_lost('transformed.csv', 'aienhancedanalytics/seeds/article.csv')
    run_dbt_build()

if __name__ == '__main__':
    main()