# Dataset

This project uses the **Customer Support on Twitter (TWCS)** dataset by Thought Vector.

Dataset source:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data

Download the dataset and place the raw CSV here:

data/twcs.csv

The raw dataset is intentionally excluded from Git because of its large file size.

Expected columns:

- tweet_id
- author_id
- inbound
- created_at
- text
- response_tweet_id
- in_response_to_tweet_id

The preprocessing pipeline reconstructs AmazonHelp conversations from the raw tweet graph and generates the derived retrieval/evidence datasets under output/.
