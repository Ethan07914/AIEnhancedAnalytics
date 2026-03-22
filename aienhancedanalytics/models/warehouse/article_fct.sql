{{
config(
        materialized='incremental'
      )
}}

SELECT
       article_pk,
       author,
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
       END AS is_sentiment_high_confidence,
       current_timestamp() as loaded_at
FROM
       {{ ref('stg_newsapi_article') }}

{% if is_incremental() %}

WHERE
      published_at > (select
                             MAX(published_at)
                      FROM
                             {{ this }}
                      )

{% endif %}