SELECT
       article_pk,
       source_name,
       title,
       published_at,
       description,
       label,
       sentiment,
       CASE
            WHEN sentiment = 'POSITIVE' THEN 1
            ELSE 0
       END AS is_positive,
       CASE
            WHEN sentiment = 'NEGATIVE' THEN 1
            ELSE 0
       END AS is_negative,
       label_probability,
       sentiment_probability,
       CASE
            WHEN label_probability > 0.5 THEN 1
            ELSE 0
       END AS is_label_high_confidence,
       CASE
            WHEN sentiment_probability > (2/3) THEN 1
            ELSE 0
       END AS is_sentiment_high_confidence
FROM
       {{ ref('stg_newsapi_article') }}