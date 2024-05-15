import re
from collections import Counter

import polars as pl

if __name__ == "__main__":
    p = re.compile(r"https?:\/\/data.cdrc.ac.uk\/search\/type\/dataset\?query=\S*")
    with open("./data/logs/url_column_accesslog_drupaldb_table_grep_search.csv") as f:
        file = f.read()
        query_matches: list[str] = p.findall(file)

    drupal_queries: list[str] = [
        query.split("=")[1]
        .lower()
        .replace("%20", " ")
        .replace("&sort_by", "")
        .replace("+", " ")
        .strip()
        for query in query_matches
    ]

    p = re.compile(r'\[.*\]\s"GET\s\/search\/type\/dataset\?query=\S*')
    with open("./data/logs/apache_access_grep_query.log") as f:
        file = f.read()
        date_matches: list[str] = p.findall(file)

    apache_queries: list[str] = [
        query.split("=")[1]
        .lower()
        .replace("%20", " ")
        .replace("&sort_by", "")
        .replace("+", " ")
        .strip()
        for query in date_matches
    ]

    drupal_queries.extend(apache_queries)
    counts = Counter(drupal_queries)

    (
        pl.DataFrame(counts)
        .transpose(include_header=True)
        .rename({"column_0": "count"})
        .sort("count", descending=True)
        .write_csv("./data/logs/queries.csv")
    )
