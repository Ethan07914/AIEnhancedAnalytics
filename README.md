# AI Enhanced Analytics

## Overview

This project showcases a complete orchestrated ETL/NLP pipeline which takes raw news data from an API and enriches the data by
assigning a sentiment and topic label to each article description which can be explored through natural language or via dashboard. 

## Natural Language Processing 

- **Text Classification:** Article descriptions where passed through the open source 'distilbert/distilbert-base-uncased-finetuned-sst-2-english' model to assign a sentiment.
- **Zero-Shot Classification:** Article descriptions and a list of labels where passed through the open source 'tasksource/deberta-small-long-nli' to assign a label.

## Conversational Analytics

[![Watch the Demo](https://res.cloudinary.com/dmeh864ji/video/upload/v1774555521/ThoughtSpot_j1uuxi.mp4)

## Dataflow
 
```mermaid
---
config:
  theme: redux-dark
---
flowchart TB
 subgraph subGraph0["ETL Pipeline"]
        B["Extract News data from NewsAPI"]
        C["Save Raw data -&gt; extracted.json"]
        D["Transform data"]
        E["Save Transformed data -> transformed.csv"]
  end
 subgraph subGraph1["ML Processing"]
        F["Sentiment Classifier Model"]
        G["Save Enriched data -> sentiment.csv"]
        H["Topic Classifier Model"]
        I["Save Enriched data -> topic.csv"]
  end
 subgraph subGraph2["Data Warehouse Modeling"]
        M["dbt Seed"]
        N["BigQuery Data Warehouse"]
        O["stg_newsapi_article"]
        P["article_fct"]
  end
 subgraph subGraph3["Analytics"]
        n2["Looker Studio Dashboard"]
        n4["Thoughtspot Conversational Analytics"]
  end
    A["Prefect Orchestration"] --> B
    B --> C
    C --> D
    D --> E
    E --> F & H & J["Join files"]
    F --> G
    H --> I
    G --> J
    I --> J
    J --> K["Load Joined data -&gt; article.csv"]
    M --> N
    N --> O
    O --> P
    P --> n2 & n4
    K --> M

```
## Tech Stack 

- **Pipeline:** Python (pandas), dbt (dbt-core, dbt-bigquery).
- **Natural Language Processing:** Transformers (Hugging Face),  PyTorch.
- **Orchestration:** Prefect Cloud.
- **Data Warehouse:** BigQuery.
- **Analytics:** ThoughtSpot, Looker Studio.

## Orchestration & CI/CD 

- **Orchestration:** Pipeline runs daily at 18:00 via Prefect Cloud.
- **CI/CD:** Prefect Cloud clones the GitHub repository before every run to ensure new changes to the pipeline are introduced.
- **Setup:** Prefect integrated well with my existing project structure, code and worked out of the box after I installed the package via pip.
- **Workflow Automation:** It was easy to connect my GitHub repo to Prefect Cloud and set up a daily cron job via CLI to automate the entire workflow.
- **Other Orchestrators:** Alternatively, Airflow could have been used; however requires additional code and would have been more costly to run via Google Cloud Composer.

![prefect_run_graph.png](prefect_run_graph.png)

## Dashboard 

- **Dashboard:** A public three-page dashboard was created with Looker Studio 
- **Link:** https://lookerstudio.google.com/u/1/reporting/34403f0c-3597-4d2d-a69d-7af99cfbe5cf/page/IdtsF

![dahboard_page1.png](dahboard_page1.png)

![dashboard_page2.png](dashboard_page2.png)

![dashboard_page3.png](dashboard_page3.png)
