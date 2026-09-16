import json
import re
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "amazonhelp_conversations.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "d2_diagnostic_examples.txt"
)


# ============================================================
# CANDIDATE ACTION WORDS
# ============================================================
#
# We are NOT saying these are actions yet.
#
# We are simply searching the real dataset for messages
# containing these words so we can inspect how AmazonHelp
# actually uses them.
# ============================================================

ACTION_WORDS = [
    "check",
    "checked",
    "checking",

    "review",
    "reviewed",
    "reviewing",

    "investigate",
    "investigated",
    "investigating",

    "refund",
    "refunded",
    "refunds",

    "process",
    "processed",
    "processing",

    "update",
    "updated",
    "updating",

    "replace",
    "replaced",
    "replacing",

    "cancel",
    "cancelled",
    "canceled",
    "cancelling",

    "verify",
    "verified",
    "verifying",

    "reset",
    "resetting",

    "resend",
    "resent",
    "resending",

    "escalate",
    "escalated",
    "escalating",

    "confirm",
    "confirmed",
    "confirming",

    "contact",
    "contacted",
    "contacting",

    "resolve",
    "resolved",
    "resolving",

    "fix",
    "fixed",
    "fixing",

    "send",
    "sent",
    "sending",

    "issue",
    "issued",

    "credit",
    "credited",

    "restore",
    "restored",

    "activate",
    "activated",
]


# ============================================================
# HELPER
# ============================================================

def normalize(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_action_word(text):

    normalized = normalize(text)

    for word in ACTION_WORDS:

        pattern = rf"\b{re.escape(word)}\b"

        if re.search(
            pattern,
            normalized,
        ):

            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("D2 DIAGNOSTIC — REAL AMAZONHELP RESPONSES")
    print("=" * 70)

    print(
        f"\nInput:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    random.seed(42)

    action_examples = []

    all_agent_examples = []

    total_agent_messages = 0

    action_message_count = 0

    # --------------------------------------------------------
    # Read reconstructed conversations
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as infile:

        for line in infile:

            conversation = json.loads(
                line
            )

            for tweet in conversation["tweets"]:

                # Only AmazonHelp messages
                if (
                    tweet["inbound"]
                    or tweet["author_id"] != "AmazonHelp"
                ):
                    continue

                text = (
                    tweet.get("text")
                    or ""
                ).strip()

                if not text:
                    continue

                total_agent_messages += 1

                record = {
                    "conversation_id":
                        conversation[
                            "conversation_id"
                        ],

                    "text":
                        text,
                }

                all_agent_examples.append(
                    record
                )

                # ------------------------------------------------
                # Search for action-related vocabulary
                # ------------------------------------------------

                if contains_action_word(text):

                    action_message_count += 1

                    action_examples.append(
                        record
                    )

    # --------------------------------------------------------
    # Randomly sample action-containing messages
    # --------------------------------------------------------

    sample_size = min(
        120,
        len(action_examples),
    )

    sampled_actions = random.sample(
        action_examples,
        sample_size,
    )

    # --------------------------------------------------------
    # Randomly sample ordinary AmazonHelp responses
    # --------------------------------------------------------

    ordinary_size = min(
        30,
        len(all_agent_examples),
    )

    sampled_all = random.sample(
        all_agent_examples,
        ordinary_size,
    )

    # --------------------------------------------------------
    # Save diagnostic report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as outfile:

        outfile.write(
            "=" * 80
            + "\n"
        )

        outfile.write(
            "D2 DIAGNOSTIC REPORT\n"
        )

        outfile.write(
            "=" * 80
            + "\n\n"
        )

        outfile.write(
            f"Total AmazonHelp messages : "
            f"{total_agent_messages:,}\n"
        )

        outfile.write(
            f"Messages containing an action word : "
            f"{action_message_count:,}\n"
        )

        if total_agent_messages:

            rate = (
                action_message_count
                / total_agent_messages
                * 100
            )

            outfile.write(
                f"Action-word message rate : "
                f"{rate:.2f}%\n"
            )

        outfile.write(
            "\n\n"
        )

        # ====================================================
        # ACTION-WORD EXAMPLES
        # ====================================================

        outfile.write(
            "=" * 80
            + "\n"
        )

        outfile.write(
            "SECTION 1 — MESSAGES CONTAINING ACTION WORDS\n"
        )

        outfile.write(
            "=" * 80
            + "\n\n"
        )

        for index, record in enumerate(
            sampled_actions,
            start=1,
        ):

            outfile.write(
                f"[{index}]\n"
            )

            outfile.write(
                f"Conversation ID: "
                f"{record['conversation_id']}\n"
            )

            outfile.write(
                f"Agent message:\n"
                f"{record['text']}\n"
            )

            outfile.write(
                "\n"
                + "-" * 80
                + "\n\n"
            )

        # ====================================================
        # ORDINARY EXAMPLES
        # ====================================================

        outfile.write(
            "=" * 80
            + "\n"
        )

        outfile.write(
            "SECTION 2 — RANDOM AMAZONHELP MESSAGES\n"
        )

        outfile.write(
            "=" * 80
            + "\n\n"
        )

        for index, record in enumerate(
            sampled_all,
            start=1,
        ):

            outfile.write(
                f"[{index}]\n"
            )

            outfile.write(
                f"Conversation ID: "
                f"{record['conversation_id']}\n"
            )

            outfile.write(
                f"Agent message:\n"
                f"{record['text']}\n"
            )

            outfile.write(
                "\n"
                + "-" * 80
                + "\n\n"
            )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"\nTotal AmazonHelp messages : "
        f"{total_agent_messages:,}"
    )

    print(
        f"Messages containing action words : "
        f"{action_message_count:,}"
    )

    if total_agent_messages:

        print(
            f"Action-word message rate : "
            f"{action_message_count / total_agent_messages * 100:.2f}%"
        )

    print(
        f"\nSampled action messages : "
        f"{sample_size}"
    )

    print(
        f"Sampled ordinary messages : "
        f"{ordinary_size}"
    )

    print(
        f"\nDiagnostic report saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()