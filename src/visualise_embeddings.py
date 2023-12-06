import os

import nomic
import numpy as np
import pinecone
from nomic import atlas

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")
nomic.login(os.environ.get("NOMIC_API_KEY"))

index = pinecone.Index("cdrc")
ids = [
    result["id"] for result in index.query([0] * 768, top_k=10000).to_dict()["matches"]
][:1000]
result = index.fetch(ids=ids)
result.to_dict()["vectors"].keys()
ids = [r for r in result.to_dict()["vectors"].keys()]
embeddings = np.array([r["values"] for r in result.to_dict()["vectors"].values()])
titles = np.array(
    [r["metadata"]["title"] for r in result.to_dict()["vectors"].values()]
)


atlas.map_embeddings(
    embeddings=embeddings,
    data=[{"id": id, "title": title} for id, title in zip(ids, titles)],
    id_field="id",
    name="cdrc",
    reset_project_if_exists=True,
)
