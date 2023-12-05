import os

from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import (
    EmbeddingRetriever,
    JoinDocuments,
    PromptNode,
    PromptTemplate,
    SentenceTransformersRanker,
    TfidfRetriever,
)
from haystack.pipelines import DocumentSearchPipeline, Pipeline


def semantic_search(query: str):
    document_store = PineconeDocumentStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment="gcp-starter",
        similarity="dot_product",
        embedding_dim=768,
        index="cdrc",
        recreate_index=False,
    )
    dense_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        use_gpu=True,
    )

    pipeline = Pipeline()
    pipeline.add_node(
        component=dense_retriever, name="DenseRetriever", inputs=["Query"]
    )
    return pipeline.run(query=query)


def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["title"], "\t", doc.score)
        print(doc.content)
        print("\n", "\n")


#
# docs = semantic_search(query="diabetes")
# pretty_print_results(docs)
#
# promp_node = PromptNode()
# prompt_template = PromptTemplate(
#     prompt="""
# You are a data analyst at the CDRC. You have been asked to answer the following query:
#
# Query: {query}
#
# You then returned the following datasets as relevant to the query:
#
# {'\n\n'.join([('Title:' + doc['meta']['title'] + '\n\n' + doc['meta']['summary']) for doc in docs])}
#
# Please explain in detail why these datasets are relevant to the query.
# """
# )
#
# out = PromptNode(
#     default_prompt_template=prompt_template,
#     model_name_or_path="google/flan-t5-large",
#     max_length=512,
# )
# docs = [doc.to_dict() for doc in tmp["documents"]]
# out(docs=docs, query=query)
