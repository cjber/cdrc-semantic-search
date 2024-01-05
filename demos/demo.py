import polars as pl
from tqdm import tqdm

from src.common.utils import Settings
from src.model import LlamaIndexModel

queries = pl.read_csv("./data/logs/queries.csv")
top = queries.head(9)[1:]["column"].to_list()


model = LlamaIndexModel(
    **Settings().model.model_dump(),
    **Settings().shared.model_dump(),
)

responses = []
for q in top:
    model.run(q, use_llm=False)
    responses.append([out["title"] for out in model.response])

compare = {
    "query": top,
    "kw": [
        ["Index of Multiple Deprivation (IMD)"],
        [
            "Access to Healthy Assets & Hazards (AHAH) (Previous Versions)",
            "Advanced GIS Methods Training: AHAH and Multi-Dimensional Indices",
            "Access to Healthy Assets & Hazards (AHAH)",
            "The Ageing in Place Classification (AiPC)",
        ],
        [
            "Index of Multiple Deprivation (IMD)",
            "CDRC Residential Mobility and Deprivation (RMD) Index (LAD Geography)",
            "CDRC Residential Mobility and Deprivation (RMD) Index (LSOA Geography)",
            "Retail Centre Boundaries and Open Indicators",
            "Advanced GIS Methods Training: AHAH and Multi-Dimensional Indices Open",
        ],
        [
            "Spatial Signatures of Great Britain",
            "London OAC",
            "Geographic Data Science in Python",
            "Local Data Spaces",
            "Linked Consumer Registers",
        ],
        [
            "Retail Centre Boundaries and Open Indicators",
            "Retail Centre Boundaries (Previous Versions)",
            "Legacy Datasets",
            "Advanced GIS Methods Training: Retail Centres and Catchment Areas",
            "Local Data Company - Retail Type or Vacancy Classification",
        ],
        [
            "Index of Multiple Deprivation (IMD)",
            "Dwelling Ages and Prices",
            "West Midlands Accessibility and Travel Passes",
            "CDRC Distances of Residential Moves (DoRM) Index (LSOA Geography)",
            "CDRC Modelled Ethnicity Proportions (LSOA Geography)",
        ],
        [
            "London OAC (2011)",
            "London Workplace Zone Classification",
            "London OAC",
            "Index of Multiple Deprivation (IMD)",
            "Creating a Geodemographic Classification Using K-means Clustering in R",
        ],
        [
            "Local morbidity rates of Global Burden of Disease and alcohol-related conditions",
            "Access to Healthy Assets & Hazards (AHAH) (Previous Versions)",
            "Access to Healthy Assets & Hazards (AHAH)",
            "UK Women's Cohort Questionnaire Data",
            "Local Data Spaces",
        ],
    ],
    "llama": responses,
}

pl.DataFrame(compare).filter(pl.col("query") == "deprivation")["kw"].to_list()
pl.DataFrame(compare).filter(pl.col("query") == "deprivation")["llama"].to_list()
