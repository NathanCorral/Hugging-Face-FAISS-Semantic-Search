import os
import math
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from datasets import Dataset


def fetch_issues(
    headers,
    logger,
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_file=None,
):
    """
    Query the Github AIP for repository issues
    """
    # Default data file (is returned)
    if issues_file is None:
        issues_file = f"./{owner}-{repo}-issues.jsonl"

    # Create directory
    data_dir = Path(os.path.dirname(issues_file))
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Github REST API Interface, download data from endpoint
    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    logger.info(f"Fetching Data ...")
    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            logger.info(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    # Create a dataset and save it to disk
    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(issues_file, orient="records", lines=True)
    logger.info(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_file}"
    )
    return issues_file


#
# Dataset loading & Filtering Functions
#
def remove_pull_and_blank_posts(dataset, 
                                columns_to_keep=["title", "body", 
                                        "html_url", "comments"]):
    """
    Remove pull requests (is_pull_request == true)
    Remove posts w/o comments (len(x["comments"]) == 0)
    Only keep selected columns
    """
    dataset = dataset.filter(
        lambda x: ((x["is_pull_request"] == False) and 
                            (len(x["comments"]) > 0) and
                            (len(x["title"]) > 0) and
                            (x["body"] is not None)) # Required  
    )

    # Remove columns
    columns = dataset.column_names
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset

def explode_column(dataset, column="comments"):
    """
    Create a separate entry in the dataset for each comment in the list of comments.
    This is equivlanat to the dataframe.explode() function
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html
    """
    dataset.set_format("pandas")
    df = dataset[:]
    comments_df = df.explode(column, ignore_index=True)
    dataset = Dataset.from_pandas(comments_df)
    return dataset