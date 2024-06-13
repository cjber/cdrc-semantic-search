import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from src.common.utils import Settings
from src.model import LlamaIndexModel

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_rows(4)

settings = Settings().model.model_dump()
settings["top_k"] = 5  # reduce eval time

model = LlamaIndexModel(**settings, load_model=True)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model.model = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},
    device_map="auto",
)
model.build_index()


past_queries = (
    pl.read_csv("data/logs/queries.csv").filter(pl.col("column") != "").head(100)
)

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
alpha_values = [0.0, 0.75, 1.0]

results = []
for alpha in tqdm(alpha_values):
    for query in tqdm(queries):
        query
        model.alpha = alpha
        model.run(query)
        evaluator = RelevancyEvaluator(service_context=model.service_context)
        contexts = [node.get_content() for node in model.response]
        eval_result = evaluator.evaluate(
            query=query,
            contexts=contexts,
            response="",
        )
        results.append({"result": eval_result.passing, "alpha": alpha, "query": query})

df = pl.DataFrame(results).with_columns(
    pl.col("alpha").cast(str), pl.col("result").cast(str)
)
df.write_csv("data/evaluation/evaluation.csv")
df = pl.read_csv("data/evaluation/evaluation.csv").with_columns(
    pl.col("alpha").cast(str), pl.col("result").cast(str)
)

sns.histplot(
    data=df,
    x="alpha",
    hue="result",
    multiple="stack",
    shrink=0.8,
    palette="gray",
)
plt.save("./data/evaluation/plot.png")
