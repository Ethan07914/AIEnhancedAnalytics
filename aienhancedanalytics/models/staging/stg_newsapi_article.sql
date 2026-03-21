SELECT
       {{ dbt_utils.generate_surrogate_key(['title', 'author', 'published_at']) }} as article_pk,
       source_name,
       title,
       author,
       published_at,
       description,
       label,
       sentiment,
       label_probability,
       sentiment_probability,
       current_timestamp() as ingested_at
from
       {{ source('articles', 'article') }}
