from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DETAILS_PATH = ROOT / "output" / "human_alignment_leave_one_out_details.csv"
CORPUS_PATH = ROOT / "output" / "retrieval_corpus.jsonl"

OUTPUT_PATH = ROOT / "output" / "failure_analysis_candidates.csv"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def text(value):
    if value is None:
        return ""

    return str(value).replace("\n", " ").strip()


def main():

    print("=" * 80)
    print("HIVER — FAILURE ANALYSIS CANDIDATE EXTRACTION")
    print("=" * 80)

    details = pd.read_csv(DETAILS_PATH)

    corpus = load_jsonl(CORPUS_PATH)

    corpus_df = pd.DataFrame(corpus)

    corpus_df["conversation_id"] = (
        corpus_df["conversation_id"]
        .astype(str)
        .str.strip()
    )

    details["conversation_id"] = (
        details["conversation_id"]
        .astype(str)
        .str.strip()
    )

    details["selected_conversation_id"] = (
        details["selected_conversation_id"]
        .astype(str)
        .str.strip()
    )

    print(f"\nEvaluation rows : {len(details):,}")
    print(f"Corpus records  : {len(corpus_df):,}")

    # --------------------------------------------------------
    # Pivot policies so each human example becomes one row
    # --------------------------------------------------------

    pivot = details.pivot(
        index=[
            "annotation_id",
            "conversation_id",
            "human_evidence_complete",
            "query_intent",
            "query_state",
        ],
        columns="policy",
        values=[
            "selected_conversation_id",
            "selected_semantic_similarity",
            "selected_ec_score",
            "selected_intent_score",
            "selected_state_score",
            "selected_evidence_tier",
        ],
    )

    pivot.columns = [
        f"{a}_{b}"
        for a, b in pivot.columns
    ]

    pivot = pivot.reset_index()

    # --------------------------------------------------------
    # Join original human query
    # --------------------------------------------------------

    original = corpus_df[
        [
            "conversation_id",
            "customer_query",
            "agent_response",
            "conversation_text",
            "turn_count",
            "evidence_completeness_score",
            "evidence_tier",
        ]
    ].copy()

    original = original.rename(
        columns={
            "customer_query": "original_customer_query",
            "agent_response": "original_agent_response",
            "conversation_text": "original_conversation_text",
            "turn_count": "original_turn_count",
            "evidence_completeness_score": "original_ec",
            "evidence_tier": "original_tier",
        }
    )

    pivot = pivot.merge(
        original,
        on="conversation_id",
        how="left",
    )

    # --------------------------------------------------------
    # Join selected conversation details for each policy
    # --------------------------------------------------------

    for policy in [
        "semantic_only",
        "semantic_intent",
        "semantic_ec",
        "full",
    ]:

        selected = corpus_df[
            [
                "conversation_id",
                "customer_query",
                "agent_response",
                "conversation_text",
                "turn_count",
                "evidence_completeness_score",
                "evidence_tier",
            ]
        ].copy()

        selected = selected.rename(
            columns={
                "conversation_id": f"{policy}_conversation_id_join",
                "customer_query": f"{policy}_customer_query",
                "agent_response": f"{policy}_agent_response",
                "conversation_text": f"{policy}_conversation_text",
                "turn_count": f"{policy}_turn_count",
                "evidence_completeness_score": f"{policy}_actual_ec",
                "evidence_tier": f"{policy}_actual_tier",
            }
        )

        pivot = pivot.merge(
            selected,
            left_on=f"selected_conversation_id_{policy}",
            right_on=f"{policy}_conversation_id_join",
            how="left",
        )

    # --------------------------------------------------------
    # Create useful disagreement / failure signals
    # --------------------------------------------------------

    pivot["ec_gain_semantic_ec"] = (
        pivot["selected_ec_score_semantic_ec"]
        -
        pivot["selected_ec_score_semantic_only"]
    )

    pivot["ec_gain_full"] = (
        pivot["selected_ec_score_full"]
        -
        pivot["selected_ec_score_semantic_only"]
    )

    pivot["semantic_loss_ec"] = (
        pivot["selected_semantic_similarity_semantic_only"]
        -
        pivot["selected_semantic_similarity_semantic_ec"]
    )

    pivot["intent_changes_selection"] = (
        pivot["selected_conversation_id_semantic_only"]
        !=
        pivot["selected_conversation_id_semantic_intent"]
    )

    pivot["ec_changes_selection"] = (
        pivot["selected_conversation_id_semantic_only"]
        !=
        pivot["selected_conversation_id_semantic_ec"]
    )

    pivot["full_changes_selection"] = (
        pivot["selected_conversation_id_semantic_ec"]
        !=
        pivot["selected_conversation_id_full"]
    )

    # --------------------------------------------------------
    # Candidate categories
    # --------------------------------------------------------

    # 1. EC strongly improves evidence but sacrifices similarity.
    pivot["candidate_ec_tradeoff"] = (
        (pivot["ec_gain_semantic_ec"] >= 0.15)
        &
        (pivot["semantic_loss_ec"] >= 0.05)
    )

    # 2. Full system loses evidence relative to EC-only.
    pivot["candidate_full_tradeoff"] = (
        (pivot["selected_ec_score_semantic_ec"]
         -
         pivot["selected_ec_score_full"] >= 0.10)
    )

    # 3. Intent changes selection but evidence doesn't improve much.
    pivot["candidate_intent_tradeoff"] = (
        pivot["intent_changes_selection"]
        &
        (pivot["ec_gain_semantic_ec"] < 0.10)
    )

    # 4. Original human example is complete, but retrieved evidence
    #    is LOW under semantic-only.
    pivot["candidate_complete_missed"] = (
        (pivot["human_evidence_complete"].str.upper() == "YES")
        &
        (pivot["selected_evidence_tier_semantic_only"] == "LOW")
    )

    # 5. Original human example is incomplete, but EC retrieval
    #    strongly favors HIGH evidence.
    pivot["candidate_incomplete_high"] = (
        (pivot["human_evidence_complete"].str.upper() == "NO")
        &
        (pivot["selected_evidence_tier_semantic_ec"] == "HIGH")
    )

    # 6. EC and full select different evidence.
    pivot["candidate_ec_full_disagreement"] = (
        pivot["selected_conversation_id_semantic_ec"]
        !=
        pivot["selected_conversation_id_full"]
    )

    # --------------------------------------------------------
    # Candidate priority score
    # --------------------------------------------------------

    pivot["candidate_score"] = 0.0

    pivot.loc[
        pivot["candidate_ec_tradeoff"],
        "candidate_score"
    ] += 3

    pivot.loc[
        pivot["candidate_full_tradeoff"],
        "candidate_score"
    ] += 2

    pivot.loc[
        pivot["candidate_intent_tradeoff"],
        "candidate_score"
    ] += 2

    pivot.loc[
        pivot["candidate_complete_missed"],
        "candidate_score"
    ] += 2

    pivot.loc[
        pivot["candidate_incomplete_high"],
        "candidate_score"
    ] += 2

    pivot.loc[
        pivot["candidate_ec_full_disagreement"],
        "candidate_score"
    ] += 1

    # --------------------------------------------------------
    # Keep interesting candidates
    # --------------------------------------------------------

    candidates = pivot[
        pivot["candidate_score"] > 0
    ].copy()

    candidates = candidates.sort_values(
        [
            "candidate_score",
            "ec_gain_semantic_ec",
        ],
        ascending=[False, False],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print concise candidate list
    # --------------------------------------------------------

    print(
        f"\nInteresting candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Saved:\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 80)
    print("TOP 25 FAILURE ANALYSIS CANDIDATES")
    print("=" * 80)

    display_columns = [
        "annotation_id",
        "conversation_id",
        "human_evidence_complete",
        "query_intent",
        "query_state",
        "original_customer_query",
        "candidate_score",
        "selected_conversation_id_semantic_only",
        "selected_ec_score_semantic_only",
        "selected_evidence_tier_semantic_only",
        "selected_conversation_id_semantic_ec",
        "selected_ec_score_semantic_ec",
        "selected_evidence_tier_semantic_ec",
        "selected_conversation_id_full",
        "selected_ec_score_full",
        "selected_evidence_tier_full",
    ]

    available = [
        col
        for col in display_columns
        if col in candidates.columns
    ]

    print(
        candidates[
            available
        ]
        .head(25)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("CANDIDATE COUNTS")
    print("=" * 80)

    print(
        "EC tradeoff              :",
        int(candidates["candidate_ec_tradeoff"].sum()),
    )

    print(
        "Full-system tradeoff     :",
        int(candidates["candidate_full_tradeoff"].sum()),
    )

    print(
        "Intent tradeoff          :",
        int(candidates["candidate_intent_tradeoff"].sum()),
    )

    print(
        "Complete cases missed    :",
        int(candidates["candidate_complete_missed"].sum()),
    )

    print(
        "Incomplete + HIGH EC     :",
        int(candidates["candidate_incomplete_high"].sum()),
    )

    print(
        "EC/full disagreement     :",
        int(candidates["candidate_ec_full_disagreement"].sum()),
    )

    print("\nSTATUS: COMPLETE")


if __name__ == "__main__":
    main()