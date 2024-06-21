
from query_utils import build_model

class Query():
    def __init__(self, cfg) -> None:
        # Grab values from config
        self.cfg = cfg
        self.logger = cfg.logger
        self.responses = getattr(cfg, 'responses', 5)

        # Construct the model (if not already built)
        if not hasattr(cfg, "get_embeddings"):
            cfg.model = build_model(cfg)
        self.get_embeddings = cfg.get_embeddings

        # Compute the FIASS Index on the dataset
        self.logger.info(f'Computing FAISS Index on dataset embeddings')
        self.dataset = cfg.dataset
        self.dataset.add_faiss_index(column="embeddings")

    def __call__(self, query : str):
        assert type(query) is str
        self.logger.debug(f'you asked me:  {query}')

        emb = self.get_embeddings([query]).detach().cpu().numpy()

        scores, samples = self.dataset.get_nearest_examples(
            "embeddings", emb, k=self.responses
        )
        return scores, samples



if __name__ == "__main__":
    import logging
    import pandas as pd

    from parser import parse_args
    from dataset_generation import get_dataset

    cfg, _ = parse_args()
    cfg.logger = logging.getLogger('HFGitHubIssuesLogger')

    cfg.dataset = get_dataset(cfg)
    # print("Dataset:  \n",  cfg.dataset)

    # Optional, can also be constructed in Query.__init__
    cfg.model = build_model(cfg)

    # Calculate FIASS on dataset and perform query
    q = Query(cfg)
    scores, samples = q( cfg.query )

    # Display results
    print()
    print(f"> {cfg.query}")
    print()
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()
