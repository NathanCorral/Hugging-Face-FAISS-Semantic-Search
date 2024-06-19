import argparse
import logging

def setup_logger(verbose=False):
    # Create logger
    logger = logging.getLogger('HFGitHubIssuesLogger')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('[%(levelname)s|%(asctime)s]: %(message)s')
    
    # Add formatter to console handler
    ch.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(ch)
    
    return logger

def parse_args():
    """
    Create the arg parser and return a dict of the program settings
    """
    parser = argparse.ArgumentParser(description="Perform a semantic search over GitHub issues.")

    # General settings
    parser.add_argument('--owner', type=str, default='huggingface',
                        help='Owner of the GitHub repository. Default is "huggingface". Arguments other than default are currently untested')
    parser.add_argument('--repo', type=str, default='datasets',
                        help='Name of the GitHub repository. Default is "datasets". Arguments other than default are currently untested')
    
    # Offline setting
    parser.add_argument('--offline', action='store_true', default=False,
                        help='Flag to try to download the dataset to use offline. Optionally also specify a path by using --dataset.')
    parser.add_argument('--dataset', type=str, default='lewtun/github-issues',
                        help='Download this from the Huggging Face dataset repo if offline is set to false')
    parser.add_argument('--data', type=str, default='./data',
                        help='Location to store downloaded dataset (if offline) and computed embeddings')
    # parser.add_argument('--dataset', type=str, default='lewtun/github-issues',
    #                     help='Dataset path. Default is "lewtun/github-issues" unless --offline is used, then it is "./data/{owner}_{repo}_issues".')

    # Dataset download settings
    parser.add_argument('--GITHUB_TOKEN', type=str, 
                        help='Github token ID, obtained from https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens. \n\
                        If left blank, will first check for a .env file, then check if exists in os environ, finally will prompt if downloading the dataset from github is required.')
    parser.add_argument('--num_issues', type=int, default=2_500, 
                        help='Number of issues to construct the dataset from.  WARNING, see https://github.com/huggingface/datasets/issues/3965 before changing the default value')

    # Debug settings
    parser.add_argument('--peak', action='store_true', default=False,
                        help='Flag to display a few examples from the dataset. Useful for debugging.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Display info on program operation.')

    # Model settings
    parser.add_argument('--model_ckpt', type=str, default='sentence-transformers/multi-qa-mpnet-base-dot-v1',
                        help='Model checkpoint to compute pooled embedding vector.')

    # Query settings
    parser.add_argument('--query', type=str, default='How can I load a dataset offline?',
                        help='Query to perform on the GitHub issues of the desired repository.')
    parser.add_argument('--responses', type=int, default=5,
                        help='Number of closest matches to return according to the FAISS metric and model embeddings.')

    # Device setting
    parser.add_argument('--device', type=str, default='cuda',
                        help='The torch device to run on. Default is "cuda".')

    args = parser.parse_args()


    # Check if owner or repo are not default and prompt user for confirmation
    if args.owner != 'huggingface' or args.repo != 'datasets':
        confirm = input(f"Getting issues from https://api.github.com/repos/{args.owner}/{args.repo} is untested and must proceed offline (with a downloaded dataset).  Do you want to continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            # parser.print_help()
            exit()
        args.offline = True

    # Adjust dataset default if offline is true
    if args.offline and args.dataset == 'lewtun/github-issues':
        args.dataset = f'./data/{args.owner}_{args.repo}_issues'

    # Set up logger based on verbose argument
    setup_logger(args.verbose)

    return args, parser


if __name__ == "__main__":
    settings, _ = parse_args()
    print(settings)