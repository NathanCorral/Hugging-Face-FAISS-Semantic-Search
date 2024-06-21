# Hugging-Face-FAISS-Semantic-Search
This repository implements the work from https://huggingface.co/learn/nlp-course/chapter5/6 to perform a semantic search over issues in Hugging Face Datasets [GitHub repository](https://github.com/huggingface/datasets).



## Running

Try it out by running:

```bash
python query.py --device="cuda" # Default device
```

with default parameters, this will:

- Download the dataset "lewtun/github-issues" from [here](https://huggingface.co/datasets/lewtun/github-issues).

  - see argument:  --dataset

- Save the dataset to disk, so it can be automatically reloaded offline on future runs.

  - default save location: "./data/huggingface-datasets-issues/"
  - see argument: --data

- Download a model/tokenizer for computing the dataset embeddings.

  - default model [here](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1):  "sentence-transformers/multi-qa-mpnet-base-dot-v1"
  - see argument:  --model_ckpt

- Preprocess the dataset.  

  - Including computing the model embeddings over the newly made "text" field.
  - The preprocessed dataset is saved and is automatically reloaded.
    - default save location:  "./data/huggingface-datasets-issues-emb/"

- Construct a Query() object which can:

  - Compute the FAISS index over the "embeddings" column of the dataset.
  - Calculate the model embedding of the query text.

  - Return the scores and samples of the *k* closest matches (responses).
    - See arguments: --query, --responses

- Display responses for the default query: "How can I load a dataset offline?"

  - Use the flag "--query ..." to make your own search!



To see the list of other parameters, run:

```
python query.py -h
```



## Building the dataset through Githubs API

The dataset can be manually constructed, following:  [https://huggingface.co/learn/nlp-course/chapter5/5](https://huggingface.co/learn/nlp-course/chapter5/5).

The default value for building the dataset with 2,500 issues works, but larger numbers may throw errors according to [this](https://github.com/huggingface/datasets/issues/3965).



**Getting a Github API Key**

Before starting, note that a Github access token is required to use the API and will throw an error if not available.  See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for generating your own access token.

The program can load token keys in the following way:

- Using [python dotenv](https://github.com/theskumar/python-dotenv), which creates an environment variable `GITHUB_TOKEN`

- If the environment is already set by:

  ```bash
  export GITHUB_TOKEN=#YOUR TOKEN HERE
  ```

- Passed on the command line with the argument --GITHUB_TOKEN



**Additional Arguments for running**

To manually scrape GitHub the repository issues and their comments, run with the flag:

```bash
python query.py --scrape --data="./data/issues2500" --verbose
```

- Using "--scrape" avoids the default behavior of downloading from a Hugging Face Dataset repo (configured with --dataset)
- It will still save the unprocessed issues and the dataset after preprocessing in the location specified by --data.
  - It will create an additional file "./data/issues2500/huggingface-datasets-issues.jsonl" that contains the issues without comments.  This gets automatically checked and will not be re-downloaded if it exists (unless updating).



**Updating**

Datasets are saved and then reloaded automatically (before and after preprocessing the embeddings).

To redownload the datasets, use the "--update" flag.



### Dependencies

Todo



## Future Work

- Build Gradio for interactive example (https://huggingface.co/learn/nlp-course/chapter9/1)
- Create search over stack exchange cooking dataset (https://drive.google.com/drive/folders/1ZWInkPCAZ-RjCr-zZrmH3XcLqSF3SRDe?usp=sharing)
  - Host: nathan.b.corral@gmail.com

