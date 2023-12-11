import re
from collections import Counter

import polars as pl

p = re.compile("https?:\/\/data.cdrc.ac.uk\/search\/type\/dataset\?query=\S*")
with open(
    "./data/logs/url_column_accesslog_drupaldb_table_grep_search.csv", mode="r"
) as f:
    file = f.read()
    query_matches = p.findall(file)

drupal_queries = [
    query.split("=")[1]
    .lower()
    .replace("%20", " ")
    .replace("&sort_by", "")
    .replace("+", " ")
    .strip()
    for query in query_matches
]

p = re.compile('\[.*\]\s"GET\s\/search\/type\/dataset\?query=\S*')
with open("./data/logs/apache_access_grep_query.log", mode="r") as f:
    file = f.read()
    date_matches = p.findall(file)

apache_queries = [
    query.split("=")[1]
    .lower()
    .replace("%20", " ")
    .replace("&sort_by", "")
    .replace("+", " ")
    .strip()
    for query in date_matches
]

drupal_queries.extend(apache_queries)
len(Counter(drupal_queries))
len(drupal_queries)

