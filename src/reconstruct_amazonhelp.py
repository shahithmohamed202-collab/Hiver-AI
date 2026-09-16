import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "twcs.csv"
OUTPUT_FILE = BASE_DIR / "output" / "amazonhelp_conversations.jsonl"

CHUNK_SIZE = 100_000
TARGET_BRAND = "AmazonHelp"


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        root = x

        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != x:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node

        return root

    def union(self, a, b):
        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a

        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    if pd.isna(value):
        return None
    return str(value)


def parse_id_list(value):
    """
    response_tweet_id can contain multiple comma-separated
    tweet IDs.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    result = []

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        try:
            result.append(int(float(part)))
        except ValueError:
            continue

    return result


# ============================================================
# START
# ============================================================

print("=" * 75)
print("AMAZONHELP CONVERSATION RECONSTRUCTION")
print("FULL-DATASET GRAPH RECONSTRUCTION")
print("=" * 75)

print(f"\nDataset : {DATA_FILE}")
print(f"Output  : {OUTPUT_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


# ============================================================
# STEP 1 — READ ENTIRE DATASET
# ============================================================

print("\n[1/5] Reading FULL TWCS dataset...")

tweets = {}

total_rows = 0

for chunk in tqdm(
    pd.read_csv(
        DATA_FILE,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "boolean",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "float64",
        },
        low_memory=False,
    ),
    desc="Reading chunks",
):

    total_rows += len(chunk)

    for row in chunk.itertuples(index=False):

        tweet_id = int(row.tweet_id)

        parent_id = None

        if not pd.isna(row.in_response_to_tweet_id):
            try:
                parent_id = int(row.in_response_to_tweet_id)
            except (ValueError, TypeError):
                parent_id = None

        response_ids = parse_id_list(
            row.response_tweet_id
        )

        tweets[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": clean_value(row.author_id),
            "inbound": (
                bool(row.inbound)
                if not pd.isna(row.inbound)
                else False
            ),
            "created_at": clean_value(row.created_at),
            "text": clean_value(row.text),
            "response_tweet_id": response_ids,
            "in_response_to_tweet_id": parent_id,
        }


print(f"\nTotal dataset rows : {total_rows:,}")
print(f"Tweets indexed     : {len(tweets):,}")


# ============================================================
# STEP 2 — BUILD FULL-DATASET CONVERSATION GRAPH
# ============================================================

print("\n[2/5] Building conversation graph over ALL tweets...")

uf = UnionFind()

for tweet_id in tqdm(
    tweets.keys(),
    desc="Adding tweet nodes",
):
    uf.add(tweet_id)


parent_links = 0
response_links = 0


for tweet_id, tweet in tqdm(
    tweets.items(),
    desc="Building graph links",
):

    # --------------------------------------------------------
    # Parent relationship
    # --------------------------------------------------------

    parent_id = tweet["in_response_to_tweet_id"]

    if parent_id is not None and parent_id in tweets:

        uf.union(tweet_id, parent_id)

        parent_links += 1


    # --------------------------------------------------------
    # Response relationship
    # --------------------------------------------------------

    for response_id in tweet["response_tweet_id"]:

        if response_id in tweets:

            uf.union(tweet_id, response_id)

            response_links += 1


print(f"\nValid parent links   : {parent_links:,}")
print(f"Valid response links : {response_links:,}")


# ============================================================
# STEP 3 — BUILD COMPONENTS
# ============================================================

print("\n[3/5] Reconstructing complete conversation components...")

components = defaultdict(list)

for tweet_id in tqdm(
    tweets.keys(),
    desc="Finding components",
):

    root = uf.find(tweet_id)

    components[root].append(tweet_id)


print(
    f"\nConversation components : "
    f"{len(components):,}"
)


# ============================================================
# STEP 4 — FILTER AMAZONHELP CONVERSATIONS
# ============================================================

print("\n[4/5] Filtering AmazonHelp conversations...")

amazonhelp_conversations = []

mixed_brand_conversations = 0
multiple_customer_conversations = 0
no_customer_conversations = 0

amazonhelp_components = 0


for root_id, tweet_ids in tqdm(
    components.items(),
    desc="Filtering components",
):

    conversation = [
        tweets[tweet_id]
        for tweet_id in tweet_ids
    ]


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    conversation.sort(
        key=lambda x: (
            x["created_at"] or ""
        )
    )


    # --------------------------------------------------------
    # Identify authors
    # --------------------------------------------------------

    authors = {
        tweet["author_id"]
        for tweet in conversation
        if tweet["author_id"] is not None
    }


    # --------------------------------------------------------
    # Must contain AmazonHelp
    # --------------------------------------------------------

    if TARGET_BRAND not in authors:
        continue

    amazonhelp_components += 1


    # --------------------------------------------------------
    # Identify customer tweets
    # --------------------------------------------------------

    customer_tweets = [
        tweet
        for tweet in conversation
        if tweet["inbound"]
    ]


    if not customer_tweets:

        no_customer_conversations += 1
        continue


    # --------------------------------------------------------
    # Exactly one customer
    # --------------------------------------------------------

    customer_ids = {
        tweet["author_id"]
        for tweet in customer_tweets
        if tweet["author_id"] is not None
    }


    if len(customer_ids) != 1:

        multiple_customer_conversations += 1
        continue


    # --------------------------------------------------------
    # Detect other obvious support brands
    #
    # We do NOT reject a conversation merely because another
    # non-customer account exists. We only reject accounts
    # that look like another support-brand handle.
    # --------------------------------------------------------

    other_support_accounts = []

    for author in authors:

        if author == TARGET_BRAND:
            continue

        if author is None:
            continue

        author_lower = author.lower()

        support_pattern = (
            author_lower.endswith("help")
            or author_lower.endswith("support")
            or "customer" in author_lower
            or "care" in author_lower
        )

        if support_pattern:
            other_support_accounts.append(author)


    if other_support_accounts:

        mixed_brand_conversations += 1
        continue


    # --------------------------------------------------------
    # Create conversation object
    # --------------------------------------------------------

    customer_id = list(customer_ids)[0]

    amazonhelp_conversations.append(
        {
            "conversation_id": root_id,
            "customer_id": customer_id,
            "brand": TARGET_BRAND,
            "turn_count": len(conversation),
            "tweets": conversation,
        }
    )


print(
    f"\nComponents containing AmazonHelp : "
    f"{amazonhelp_components:,}"
)

print(
    f"AmazonHelp conversations retained : "
    f"{len(amazonhelp_conversations):,}"
)

print(
    f"No-customer removed                : "
    f"{no_customer_conversations:,}"
)

print(
    f"Multiple-customer removed          : "
    f"{multiple_customer_conversations:,}"
)

print(
    f"Mixed-brand removed                 : "
    f"{mixed_brand_conversations:,}"
)


# ============================================================
# STEP 5 — SAVE JSONL
# ============================================================

print("\n[5/5] Saving reconstructed conversations...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    for conversation in amazonhelp_conversations:

        f.write(
            json.dumps(
                conversation,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# VALIDATION
# ============================================================

turn_counts = [
    conversation["turn_count"]
    for conversation in amazonhelp_conversations
]


if turn_counts:

    average_turns = (
        sum(turn_counts)
        / len(turn_counts)
    )

    conversations_4_plus = sum(
        count >= 4
        for count in turn_counts
    )

    percentage_4_plus = (
        conversations_4_plus
        / len(turn_counts)
        * 100
    )

else:

    average_turns = 0
    conversations_4_plus = 0
    percentage_4_plus = 0


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print(
    f"Raw TWCS rows          : "
    f"{total_rows:,}"
)

print(
    f"Tweets indexed         : "
    f"{len(tweets):,}"
)

print(
    f"Graph components       : "
    f"{len(components):,}"
)

print(
    f"AmazonHelp components  : "
    f"{amazonhelp_components:,}"
)

print(
    f"Final conversations    : "
    f"{len(amazonhelp_conversations):,}"
)

print(
    f"Average turns          : "
    f"{average_turns:.2f}"
)

print(
    f"4+ tweet conversations : "
    f"{conversations_4_plus:,}"
)

print(
    f"Percentage 4+ turns   : "
    f"{percentage_4_plus:.2f}%"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nDone.")