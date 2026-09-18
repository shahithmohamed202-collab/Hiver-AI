import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)


# =============================================================================
# PATHS / CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "human_verification_150.csv"

# V2 uses separate output files.
OUTPUT_CSV = BASE_DIR / "output" / "llm_judge_results_v2.csv"
OUTPUT_JSON = BASE_DIR / "output" / "llm_judge_evaluation_v2.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

TIMEOUT_SECONDS = 300
MAX_RETRIES = 3

TEMPERATURE = 0.0
NUM_PREDICT = 300

REQUEST_DELAY_SECONDS = 0.2


# =============================================================================
# JUDGE RUBRIC
# =============================================================================

SYSTEM_PROMPT = r"""
You are an expert evaluator judging customer-support conversations.

Your task is to evaluate OBSERVABLE EVIDENCE in the conversation.

VERY IMPORTANT:

Evidence completeness is NOT the same as successful problem resolution.

A conversation can be evidence-complete even if:
- the customer's issue is not fully resolved,
- the customer is still waiting,
- the customer is unhappy,
- the final outcome is unknown,
- AmazonHelp asks for specific information needed to investigate,
- AmazonHelp gives specific instructions,
- AmazonHelp provides a concrete support route,
- AmazonHelp provides meaningful troubleshooting,
- the customer confirms an observable outcome.

The goal is to determine whether the historical conversation contains enough
meaningful observable support evidence.

----------------------------------------------------------------------
EVIDENCE_COMPLETE
----------------------------------------------------------------------

Return YES when either of these exists:

A. MEANINGFUL SUPPORT ACTION

Examples:
- specific troubleshooting instructions
- specific next steps
- requesting information needed for investigation
- directing the customer to a specific support channel
- directing the customer to a specific form
- explaining a concrete operational action
- escalation/investigation
- meaningful policy or operational guidance

OR

B. USEFUL OBSERVABLE OUTCOME

Examples:
- customer confirms receiving a refund
- customer confirms receiving an item
- customer confirms something works
- customer confirms a return/pickup outcome
- customer provides a meaningful observable follow-up outcome

Evidence completeness does NOT require successful resolution.

Return NO when the conversation contains only:
- greetings
- apologies
- acknowledgements
- vague promises
- generic customer-service language
- "please DM us" with no substantive support action
- an unresolved interaction with neither meaningful action nor useful outcome

----------------------------------------------------------------------
ACTION_OBSERVED
----------------------------------------------------------------------

YES when AmazonHelp performs or clearly states a concrete support action.

YES examples:
- troubleshooting steps
- specific instructions
- asking for information required to proceed
- support phone/chat/form route
- investigation
- escalation
- meaningful operational guidance

NO examples:
- greeting only
- apology only
- acknowledgement only
- vague "we'll help" language

----------------------------------------------------------------------
CUSTOMER_REACTION
----------------------------------------------------------------------

YES only when the customer provides an observable response AFTER a relevant
AmazonHelp action.

Examples:
- "Thanks"
- "It worked"
- "Still not working"
- "I received the refund"
- follow-up information
- dissatisfaction after the action

Silence after an action = NO.

A customer response BEFORE the relevant agent action does not count.

Customer reaction does NOT mean successful resolution.

----------------------------------------------------------------------
DM_REDIRECT
----------------------------------------------------------------------

YES when AmazonHelp redirects the customer to DM/private messaging before
a concrete public support action.

A DM redirect can coexist with action and reaction.

A DM redirect by itself does NOT make evidence complete.

----------------------------------------------------------------------
EDGE CASES
----------------------------------------------------------------------

1. "Thanks" after a concrete support action:
   action = YES
   reaction = YES
   evidence may be YES

2. Customer remains unhappy after a concrete support action:
   action = YES
   reaction = YES if the response occurs afterward
   evidence may still be YES

3. AmazonHelp asks for specific information needed to investigate:
   action = YES
   evidence may be YES

4. AmazonHelp only says "please DM us":
   dm_redirect = YES
   action = NO
   evidence = NO unless another meaningful action/outcome exists

5. Customer confirms an observable outcome:
   reaction = YES when it follows relevant support interaction
   evidence may be YES

6. Never infer hidden actions, hidden outcomes, or information not visible.

7. If genuinely uncertain, choose NO.

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

Return EXACTLY ONE JSON OBJECT:

{
  "evidence_complete": "YES" or "NO",
  "action_observed": "YES" or "NO",
  "customer_reaction": "YES" or "NO",
  "dm_redirect": "YES" or "NO",
  "confidence": 0.0,
  "reason": "short explanation"
}

Do not output markdown.
Do not output multiple JSON objects.
Do not output headings.
Do not output anything outside the JSON object.
"""


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_yes_no(value):
    if value is None:
        return None

    value = str(value).strip().upper()

    if value in {"YES", "Y", "TRUE", "1"}:
        return "YES"

    if value in {"NO", "N", "FALSE", "0"}:
        return "NO"

    return None


# =============================================================================
# ROBUST JSON EXTRACTION
# =============================================================================

def extract_json(text):
    """
    Extract the first valid JSON object from model output.

    Handles:
    - normal JSON
    - markdown fences
    - leading text
    - trailing text
    - duplicated JSON responses
    """

    if not text:
        return None

    text = text.strip()

    # First: try the entire response.
    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    # Second: scan for the first valid JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):

        if character != "{":
            continue

        try:
            data, _ = decoder.raw_decode(text[index:])

            if isinstance(data, dict):
                return data

        except Exception:
            continue

    return None


# =============================================================================
# VALIDATE JUDGE OUTPUT
# =============================================================================

def validate_judge_output(data):

    if not isinstance(data, dict):
        return None

    evidence = normalize_yes_no(
        data.get("evidence_complete")
    )

    action = normalize_yes_no(
        data.get("action_observed")
    )

    reaction = normalize_yes_no(
        data.get("customer_reaction")
    )

    dm = normalize_yes_no(
        data.get("dm_redirect")
    )

    if evidence is None:
        return None

    if action is None:
        return None

    if reaction is None:
        return None

    if dm is None:
        return None

    try:
        confidence = float(
            data.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    reason = str(
        data.get("reason", "")
    ).strip()

    return {
        "evidence_complete": evidence,
        "action_observed": action,
        "customer_reaction": reaction,
        "dm_redirect": dm,
        "confidence": confidence,
        "reason": reason,
    }


# =============================================================================
# BUILD PROMPT
# =============================================================================

def build_prompt(row, repair=False):

    conversation = row.get(
        "conversation_text",
        ""
    ).strip()

    if not conversation:

        customer_messages = row.get(
            "customer_messages",
            ""
        )

        agent_messages = row.get(
            "agent_messages",
            ""
        )

        conversation = (
            "CUSTOMER MESSAGES:\n"
            f"{customer_messages}\n\n"
            "AMAZONHELP MESSAGES:\n"
            f"{agent_messages}"
        )

    repair_text = ""

    if repair:

        repair_text = r"""

Your previous response was invalid.

Return ONLY one valid JSON object.

Required keys:

evidence_complete
action_observed
customer_reaction
dm_redirect
confidence
reason

Do not output markdown.
Do not output multiple objects.
Do not output explanations outside JSON.
"""

    return f"""
{SYSTEM_PROMPT}

Now evaluate this customer-support conversation.

================ CONVERSATION ================

{conversation}

============== END CONVERSATION ==============

Remember:

- Evidence completeness is NOT successful resolution.
- A meaningful support action can make evidence complete.
- A useful observable outcome can make evidence complete.
- Generic apologies and greetings are insufficient.
- A DM redirect alone is insufficient.
- Customer reaction must occur after relevant support action.
- Do not infer hidden actions or outcomes.

{repair_text}

Return exactly one JSON object.
"""


# =============================================================================
# OLLAMA CALL
# =============================================================================

def call_ollama(prompt):

    last_error = ""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"  Ollama attempt "
            f"{attempt}/{MAX_RETRIES}..."
        )

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "seed": 42,
            },
            "keep_alive": "5m",
        }

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(
                payload
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=TIMEOUT_SECONDS
            ) as response:

                raw_response = response.read().decode(
                    "utf-8"
                )

            body = json.loads(
                raw_response
            )

            model_text = body.get(
                "response",
                ""
            )

            parsed = extract_json(
                model_text
            )

            validated = validate_judge_output(
                parsed
            )

            if validated is not None:
                return validated

            print(
                "  Invalid JSON/output; "
                "retrying with stricter instruction..."
            )

            prompt = build_prompt_from_existing(
                prompt
            )

            last_error = (
                "Invalid or incomplete judge output."
            )

        except urllib.error.URLError as error:

            last_error = str(error)

            print(
                f"  Ollama connection error: "
                f"{error}"
            )

        except Exception as error:

            last_error = str(error)

            print(
                f"  Judge processing error: "
                f"{error}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(1)

    raise RuntimeError(
        "LLM judge failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def build_prompt_from_existing(prompt):

    return (
        prompt
        + """

FINAL JSON REQUIREMENT:

Return exactly one JSON object.

Start with {.
End with }.

No markdown.
No explanation.
No second JSON object.
"""
    )


# =============================================================================
# CSV FUNCTIONS
# =============================================================================

RESULT_FIELDS = [
    "annotation_id",
    "conversation_id",
    "human_evidence_complete",
    "judge_evidence_complete",
    "judge_action_observed",
    "judge_customer_reaction",
    "judge_dm_redirect",
    "judge_confidence",
    "judge_reason",
    "agreement",
]


def load_annotations():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(
            csv.DictReader(file)
        )


def load_checkpoint():

    if not OUTPUT_CSV.exists():
        return {}

    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    checkpoint = {}

    for row in rows:

        annotation_id = str(
            row.get(
                "annotation_id",
                ""
            )
        ).strip()

        if annotation_id:
            checkpoint[annotation_id] = row

    return checkpoint


def save_results(results):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS
        )

        writer.writeheader()

        for row in results:

            writer.writerow({
                field: row.get(
                    field,
                    ""
                )
                for field in RESULT_FIELDS
            })


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(results, human_annotations):

    # -------------------------------------------------------------------------
    # Evidence-completeness agreement
    # -------------------------------------------------------------------------

    human = [
        1
        if row["human_evidence_complete"] == "YES"
        else 0
        for row in results
    ]

    judge = [
        1
        if row["judge_evidence_complete"] == "YES"
        else 0
        for row in results
    ]

    accuracy = accuracy_score(
        human,
        judge
    )

    precision = precision_score(
        human,
        judge,
        zero_division=0
    )

    recall = recall_score(
        human,
        judge,
        zero_division=0
    )

    f1 = f1_score(
        human,
        judge,
        zero_division=0
    )

    kappa = cohen_kappa_score(
        human,
        judge
    )

    cm = confusion_matrix(
        human,
        judge,
        labels=[0, 1]
    )

    # -------------------------------------------------------------------------
    # Build human annotation lookup
    #
    # IMPORTANT:
    # The human CSV contains:
    #
    # action_observed
    # customer_reaction_observed
    # dm_redirect
    #
    # while judge results contain:
    #
    # judge_action_observed
    # judge_customer_reaction
    # judge_dm_redirect
    #
    # We explicitly map these fields below.
    # -------------------------------------------------------------------------

    human_lookup = {}

    for human_row in human_annotations:

        annotation_id = str(
            human_row.get(
                "annotation_id",
                ""
            )
        ).strip()

        if annotation_id:
            human_lookup[annotation_id] = human_row

    # -------------------------------------------------------------------------
    # Criterion-level agreement
    # -------------------------------------------------------------------------

    criterion_pairs = [
        (
            "action_observed",
            "judge_action_observed"
        ),
        (
            "customer_reaction_observed",
            "judge_customer_reaction"
        ),
        (
            "dm_redirect",
            "judge_dm_redirect"
        ),
    ]

    criterion_agreement = {}

    criterion_counts = {}

    for human_field, judge_field in criterion_pairs:

        pairs = []

        for result in results:

            annotation_id = str(
                result.get(
                    "annotation_id",
                    ""
                )
            ).strip()

            human_row = human_lookup.get(
                annotation_id
            )

            if human_row is None:
                continue

            human_value = normalize_yes_no(
                human_row.get(
                    human_field
                )
            )

            judge_value = normalize_yes_no(
                result.get(
                    judge_field
                )
            )

            if (
                human_value is not None
                and judge_value is not None
            ):

                pairs.append(
                    (
                        human_value,
                        judge_value
                    )
                )

        if pairs:

            agreement = sum(
                human_value == judge_value
                for human_value, judge_value in pairs
            ) / len(pairs)

        else:

            agreement = 0.0

        criterion_agreement[
            human_field
        ] = agreement

        criterion_counts[
            human_field
        ] = len(pairs)

    return {
        "examples": len(results),

        "human_yes": sum(human),
        "human_no": len(human) - sum(human),

        "judge_yes": sum(judge),
        "judge_no": len(judge) - sum(judge),

        "exact_agreement": accuracy,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "cohen_kappa": kappa,

        "criterion_agreement": criterion_agreement,
        "criterion_evaluated_counts": criterion_counts,

        "confusion_matrix": cm.tolist(),
    }


def save_metrics(metrics):

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "HIVER — LLM JUDGE / HUMAN AGREEMENT V2"
    )
    print("=" * 80)

    print()
    print(
        f"Input file             : {INPUT_FILE}"
    )
    print(
        f"Ollama endpoint        : {OLLAMA_URL}"
    )
    print(
        f"Model                  : {MODEL_NAME}"
    )
    print(
        f"Temperature            : {TEMPERATURE}"
    )
    print(
        "JSON mode              : enabled"
    )
    print(
        "Judge rubric            : V2 calibrated"
    )
    print(
        f"Output CSV             : {OUTPUT_CSV}"
    )
    print(
        f"Output JSON            : {OUTPUT_JSON}"
    )
    print()

    rows = load_annotations()

    print(
        f"Human examples loaded  : {len(rows)}"
    )

    checkpoint = load_checkpoint()

    if checkpoint:

        print(
            f"V2 checkpoint found    : "
            f"{len(checkpoint)}"
        )

    print()

    results = []

    for index, row in enumerate(
        rows,
        start=1
    ):

        annotation_id = str(
            row.get(
                "annotation_id",
                ""
            )
        ).strip()

        conversation_id = str(
            row.get(
                "conversation_id",
                ""
            )
        ).strip()

        if annotation_id in checkpoint:

            results.append(
                checkpoint[annotation_id]
            )

            print(
                f"[{index:03d}/{len(rows)}] "
                f"annotation={annotation_id} "
                f"CHECKPOINT — SKIPPED"
            )

            continue

        print(
            f"[{index:03d}/{len(rows)}] "
            f"annotation={annotation_id} "
            f"conversation={conversation_id}"
        )

        prompt = build_prompt(row)

        judge = call_ollama(
            prompt
        )

        human_label = normalize_yes_no(
            row.get(
                "evidence_complete"
            )
        )

        if human_label is None:

            raise ValueError(
                "Invalid human evidence label "
                f"for annotation {annotation_id}: "
                f"{row.get('evidence_complete')}"
            )

        agreement = (
            "AGREE"
            if human_label
            == judge["evidence_complete"]
            else "DISAGREE"
        )

        result = {
            "annotation_id": annotation_id,
            "conversation_id": conversation_id,

            "human_evidence_complete":
                human_label,

            "judge_evidence_complete":
                judge["evidence_complete"],

            "judge_action_observed":
                judge["action_observed"],

            "judge_customer_reaction":
                judge["customer_reaction"],

            "judge_dm_redirect":
                judge["dm_redirect"],

            "judge_confidence":
                judge["confidence"],

            "judge_reason":
                judge["reason"],

            "agreement":
                agreement,
        }

        results.append(result)

        save_results(
            results
        )

        print(
            f"  Human={human_label} "
            f"Judge={judge['evidence_complete']} "
            f"Confidence={judge['confidence']:.2f} "
            f"{agreement}"
        )

        print(
            f"  Checkpoint saved "
            f"({len(results)}/{len(rows)})"
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Metrics use BOTH the judge results and the original human annotations.
    # This fixes the previous criterion-agreement bug.
    # -------------------------------------------------------------------------

    metrics = calculate_metrics(
        results,
        rows
    )

    save_metrics(
        metrics
    )

    print()
    print("=" * 80)
    print("LLM JUDGE V2 RESULTS")
    print("=" * 80)

    print()

    print(
        f"Examples               : "
        f"{metrics['examples']}"
    )

    print(
        f"Human YES              : "
        f"{metrics['human_yes']}"
    )

    print(
        f"Human NO               : "
        f"{metrics['human_no']}"
    )

    print(
        f"Judge YES              : "
        f"{metrics['judge_yes']}"
    )

    print(
        f"Judge NO               : "
        f"{metrics['judge_no']}"
    )

    print()

    print(
        f"Exact agreement        : "
        f"{metrics['exact_agreement']:.4f}"
    )

    print(
        f"Accuracy               : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision              : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall                 : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                     : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Cohen's kappa          : "
        f"{metrics['cohen_kappa']:.4f}"
    )

    print()
    print("Criterion agreement:")

    criterion_labels = {
        "action_observed":
            "action_observed",

        "customer_reaction_observed":
            "customer_reaction",

        "dm_redirect":
            "dm_redirect",
    }

    for human_field, value in metrics[
        "criterion_agreement"
    ].items():

        display_name = criterion_labels.get(
            human_field,
            human_field
        )

        count = metrics[
            "criterion_evaluated_counts"
        ].get(
            human_field,
            0
        )

        print(
            f"  {display_name:22s}: "
            f"{value:.4f} "
            f"({count}/{metrics['examples']} evaluated)"
        )

    print()

    print(
        "Confusion matrix "
        "[human NO/YES × judge NO/YES]:"
    )

    for row_cm in metrics[
        "confusion_matrix"
    ]:

        print(row_cm)

    print()
    print("Outputs:")

    print(
        f"  CSV : {OUTPUT_CSV}"
    )

    print(
        f"  JSON: {OUTPUT_JSON}"
    )

    print()
    print(
        "STATUS: LLM JUDGE V2 EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()