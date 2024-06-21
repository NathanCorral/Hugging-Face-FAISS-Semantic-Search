import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests

from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

import torch

from src.dataset_utils import fetch_issues, remove_pull_and_blank_posts, explode_column


def get_dataset(settings, 
                update=None, 
                # dataset_dir=None,
                # issues_dataset_dir=None,
                # issues_json_file=None,
                ):
    

    # Default Values
    if update is None:
        update = settings.update
    dataset_dir = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues-emb"
    issues_json_file = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues.jsonl"
    issues_dataset_dir = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues"
    # if dataset_dir is None:
    #     dataset_dir = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues-emb"
    # if issues_json_file is None:
    #     issues_json_file = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues.jsonl"
    # if issues_dataset_dir is None:
    #     issues_dataset_dir = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues"
    # Create logger
    logger = logging.getLogger('HFGitHubIssuesLogger')
    
    # Try first to load directly the dataset with precomputed embeddings
    if not update and os.path.isdir(dataset_dir):
        logger.info(f'Loading pre-processed dataset from {dataset_dir}')
        dataset = load_from_disk(dataset_dir)
        return dataset

    # Otherwise the dataset possibly needs to be downloaded and constructed.
    if not update and os.path.isdir(issues_dataset_dir):
        # First try and load the already downloaded dataset from disk.  (this is dataset without preprocessing)
        logger.info(f'Loading dataset from disk:  {issues_dataset_dir}')
        issues_dataset = load_from_disk(issues_dataset_dir)
    else:
        # Need to download from scratch
        if not settings.scrape: 
            # Grab the dataset from HF repo instead of creating it ourselves
            logger.info(f'Fetching the dataset from:  {settings.dataset}')
            issues_dataset = load_dataset(settings.dataset, split="train")
            # Save for next time
            issues_dataset.save_to_disk(issues_dataset_dir)
        else:
            # Manually construct dataset from Github API
            issues_dataset = download_data(settings, logger, update,
                                            issues_json_file=issues_json_file, 
                                            issues_dataset_dir=issues_dataset_dir)

    # With the issues_dataset, we can apply pre-processing steps 
    dataset = preprocess_dataset(settings, issues_dataset, logger)
    # Save for next time
    dataset.save_to_disk(dataset_dir)
    return dataset

def download_data(settings, logger,
                    update=False,
                    issues_json_file=None,
                    issues_dataset_dir=None):
    """
    Download or update the issues and comments from a selected GitHub repo.
    :param issues_json_file:  Save the Issues in a json file.
    :param issues_dataset:    Download the comments for each issue and save them together in this dataset

    TODO:  Improve Error checking, 
            throw proper error on incorrect response (instead of assert), 
            fix issue here:  https://github.com/huggingface/datasets/issues/3965,
            Check dataset constructed by load_dataset
    """
    # Default Values
    if issues_json_file is None:
        issues_json_file = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues.jsonl"
    if issues_dataset_dir is None:
        issues_dataset_dir = f"{Path(settings.data)}/{settings.owner}-{settings.repo}-issues"

    # Test connection
    test_url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
    logger.debug(f"Checking connection to:   {test_url}")
    response = requests.get(test_url)
    logger.debug(f"Github Response Status Code:   {response.status_code}")
    assert (response.status_code == 200), f'Expected successful response status code of 200, got:  {response.status_code}'


    # Create Github Headers 
    headers = create_headers(settings)

    # Fetch issues from repo
    if update or (not os.path.isfile(issues_json_file)):
        issues_json_file = fetch_issues(headers, logger,
                        owner=settings.owner,
                        repo=settings.repo,
                        num_issues=settings.num_issues,
                        issues_file=issues_json_file,
                        )

    # Construct a dataset from the file
    issues_dataset = load_dataset("json", data_files=issues_json_file, split="train")

    # Download comments for each issue
    logger.debug(f'Downloading comments for each Issue...')
    def get_comments(issue_number):
        url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
        response = requests.get(url, headers=headers)
        return [r["body"] for r in response.json()]
    # Depending on your internet connection, this can take a few minutes...
    issues_dataset = issues_dataset.map(
        lambda x: {"comments": get_comments(x["number"])}
    )

    # Save the dataset to disk
    issues_dataset.save_to_disk(issues_dataset_dir)
    return issues_dataset

def preprocess_dataset(settings, issues_dataset, logger,
                            columns_to_keep=["title", "body", 
                                    "html_url", "comments"],
                                    min_words=15,
                                    dataset_dir=None):
    """
    Create the comments dataset by pre-processing the raw issues data downloaded from Github.
    Steps:
        1. Set flag to mark 'issues' from 'pull requests'
             Filter out pull requests and posts without comments/titles/bodies
             Remove unused columns
        2. 'Explode' the comments column so each element is duplicated
        3. Filter out comments less than min_words
        4. Concatenate the title, body, and comment into new field called "text"
        5. Computer Embeddings
    :return:  The preprocessed dataset
    """

    # 1.
    logger.debug(f'Marking and Removing pull reuqests...')
    dataset = issues_dataset.map(
        lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
    )
    dataset = remove_pull_and_blank_posts(dataset, columns_to_keep=columns_to_keep)

    # 2.
    logger.debug(f'Exploding columns...')
    dataset = explode_column(dataset, column="comments")

    # 3.
    dataset = dataset.filter(lambda x: len(x["comments"].split()) > min_words)

    # 4.
    def concatenate_text(examples):
        return {"text": examples["title"]
            + " \n "
            + examples["body"]
            + " \n "
            + examples["comments"]}

    dataset = dataset.map(concatenate_text)

    # Compute the embeddings for the dataset
    dataset = compute_dataset_embeddings(settings, dataset, logger)

    return dataset

def compute_dataset_embeddings(settings, dataset, logger):
    """
    Embed the 'text' field of the dataset into a vector containing the semantic meaning.
    This will then be compared to the querey which computes a vector with the same model.

    Helper function for compute_embeddings(..)
    """
    logger.debug("Loading model...")
    device = torch.device(settings.device)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_ckpt)
    model = AutoModel.from_pretrained(settings.model_ckpt)
    model.to(device)

    def cls_pooling(model_output):
        """
        Use the 'cls' vector from the models output as the encoding vector
        """
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)

    logger.debug("Computing Embeddings...")
    dataset = dataset.map(
        lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
    )

    return dataset


def create_headers(settings):
    """
    Create the response API header containing the github token ID.
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    """
    # 1.  Check if passed by command line
    if settings.GITHUB_TOKEN is not None:
        return {"Authorization": f"token {settings.GITHUB_TOKEN}"}
    
    # 2.  (Recommended) Load from .env if it exists
    if os.path.exists(".env"):
        load_dotenv()

    # 3. Load already set in environment
    if "GITHUB_TOKEN" in os.environ:
        return {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}

    raise RuntimeError(f'GITHUB_TOKEN should be set on environment or with .env file (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).')


if __name__ == "__main__":
    import pandas as pd
    from src.parser import parse_args
    settings, _ = parse_args()
    dataset = get_dataset(settings)
    print(dataset)

    # Display a few samples
    dataset.set_format("pandas")
    df = dataset[:]

    pd.set_option("display.max.columns", None)
    # pd.set_option("display.precision", 2)

    print(df.head())
    # print(settings)