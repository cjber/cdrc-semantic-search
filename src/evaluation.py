import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from llama_index.evaluation import RelevancyEvaluator
from tqdm import tqdm

from src.common.utils import Settings
from src.model import LlamaIndexModel

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_rows(4)

settings = Settings().model.model_dump()
settings["top_k"] = 5  # reduce cost of evaluation

model = LlamaIndexModel(**Settings().model.model_dump(), load_model=True)


past_queries = (
    pl.read_csv("data/logs/queries.csv").filter(pl.col("column") != "").head(100)
)

# TODO: Add definite fail cases?
fails = ["supercars"]  # these cases should always output 'false'
queries = [
    "social mobility",
    "mobility",
    "diabetes",
    "health",
    "liverpool",
    "london",
    "covid",
    "greenspace",
] + fails
queries.extend([f"{query} datasets" for query in queries])
queries.extend([f"datasets relating to {query}" for query in queries])
queries.extend(past_queries["column"].to_list())
alpha_values = [0.0, 0.5, 1.0]

results = []
for alpha in tqdm(alpha_values):
    for query in tqdm(queries):
        model.alpha = alpha
        model.run(query, use_llm=True)
        evaluator = RelevancyEvaluator(service_context=model.service_context)
        contexts = [node.get_content() for node in model.response.source_nodes]
        eval_result = evaluator.evaluate(
            query=query,
            contexts=contexts,
            response=model.response.response,
        )
        results.append({"result": eval_result.passing, "alpha": alpha, "query": query})

df = pl.DataFrame(results).with_columns(
    pl.col("alpha").cast(str), pl.col("result").cast(str)
)
df.write_csv("data/evaluation/evaluation.csv")
sns.histplot(
    data=df,
    x="alpha",
    hue="result",
    multiple="stack",
    shrink=0.8,
    stat="percent",
    palette="gray",
)
plt.show()
