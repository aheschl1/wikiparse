
from pathlib import Path
from wikindex.config import Config
from wikindex.data.dataset import Datapoint, Dataset, TextDatapoint
from wikindex.data.index import Index
from wikindex.custom_colbert.model import ColBertScorer, ColBert


def get_config():
    return Config(
        device='cuda',
        cache_dir="/home/andrew/Documents/wikiparse/.models"
    )

if __name__ == "__main__":
    config = get_config()
    texts = [
        "This is a test sentence. Now, it is even longer than before. fkdskl, dfksajfhkj",
        "Another test sentence for encoding.",
        "Fred hates to code.",
        "Andrew is a person who loves to test code."
    ]
    datapoints = {i: TextDatapoint(id=i, text=text) for i, text in enumerate(texts)}
    dataset = Dataset(datapoints)
    encoder = ColBert(config=config)
    print(f"Number of documents in dataset: {len(dataset)}")
    scorer = ColBertScorer(config=config)
    index = Index(dataset, encoder, scorer, config=config)
    index.build()
    print("Embedding dataset...")
    query = "Who enjoys programming?"
    scores, indices = index.search(query, top_k=2)
    print("Top scores:", scores)
    print("Top indices:", indices)