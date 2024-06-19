import os
import logging
from dotenv import load_dotenv

from datasets import load_dataset

from dataset_utils import fetch_issues



def fetch_issues(
    headers,
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):

def check_load_dataset(settings):
    # Create logger
    logger = logging.getLogger('HFGitHubIssuesLogger')

    if not settings.offline:
        logger.debug(f'Fetching the dataset from {settings.dataset}')
        issues_dataset = load_dataset(settings.dataset, split="train")
        return issues_dataset
    # else

    # Check if dataset is already downloaded
    if os.path.exists(settings.dataset):
        logger.debug(f'Dataset already downloaded.  Loading from folder {settings.dataset}')
        issues_dataset = load_from_disk(settings.dataset)
        return issues_dataset
    #else

    # Construct the dataset by fetching issues from github
    headers = create_headers(settings)
    fetch_issues(headers,
                    owner=settings.owner,
                    repo=settings.repo,
                    # num_issues=settings.num_issues,
                    num_issues=29,
                    owner=settings.owner,
                    owner=settings.owner,
                    )


def create_headers(settings):
    """
    Create the response API header containing the github token ID
    """
    # 1.  Check if passed by command line
    if settings.GITHUB_TOKEN is not None:
        return {"Authorization": f"token {settings.GITHUB_TOKEN}"}
    
    # 2.  (Recommended) Load from .env if it exists
    if os.path.exists(".env"):
        load_dotenv()

    if "GITHUB_TOKEN" in os.environ:
        return {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}

    raise RuntimeError(f'GITHUB_TOKEN should be set on environment or with .env file (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).')


if __name__ == "__main__":
    from parser import parse_args
    settings, _ = parse_args()
    check_load_dataset(settings)
    # print(settings)