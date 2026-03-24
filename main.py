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

    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"{project_dir} not found")

    command = [
        "dbt", "build",
        "--project-dir", project_dir,
        "--profiles-dir", project_dir
    ]

    result = subprocess.run(command, text=True, capture_output=True)

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"dbt build failed\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    logging.info("dbt build completed successfully!")

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