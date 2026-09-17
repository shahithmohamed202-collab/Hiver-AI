from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

SOURCE_SAMPLE = OUTPUT_DIR / "human_annotation_sample.csv"
VERIFICATION_FILE = OUTPUT_DIR / "human_verification_150.csv"

TARGET_SIZE = 150

LABEL_COLUMNS = [
    "evidence_complete",
    "action_observed",
    "customer_reaction_observed",
    "dm_redirect",
]

NOTE_COLUMN = "notes"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hiver AI — Human Verification",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
        background: #fafafa;
    }

    .conversation-box {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        background: #ffffff;
    }

    .customer {
        background: #f5f7ff;
        border-left: 4px solid #4f6bed;
    }

    .agent {
        background: #f7f7f7;
        border-left: 4px solid #777;
    }

    .small-note {
        color: #777;
        font-size: 0.85rem;
    }

    .suggestion {
        padding: 8px 12px;
        border-radius: 8px;
        background: #fff7df;
        border: 1px solid #e8d28a;
        margin-bottom: 10px;
    }

    .progress-text {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value):
    value = clean_value(value).upper()

    if value in {"YES", "Y", "TRUE", "1"}:
        return "YES"

    if value in {"NO", "N", "FALSE", "0"}:
        return "NO"

    return ""


def split_messages(value):
    value = clean_value(value)

    if not value:
        return []

    # The retrieval corpus stores turns separated by ||
    if "||" in value:
        return [x.strip() for x in value.split("||") if x.strip()]

    return [value]


def get_bool(value):
    return normalize_label(value) == "YES"


# ============================================================
# AUTOMATIC PRE-LABEL SUGGESTIONS
# ============================================================

def suggest_action_observed(row):
    """
    Suggest whether a concrete support action was observed.

    IMPORTANT:
    This is only a suggestion for human verification.
    """

    d2 = clean_value(row.get("D2_action_specificity", ""))

    try:
        score = float(d2)
    except ValueError:
        score = 0.0

    return "YES" if score >= 0.40 else "NO"


def suggest_customer_reaction(row):
    """
    Silence is NOT treated as a positive reaction.

    Positive/negative/neutral observable reactions are suggested YES.
    Silence/no-action is suggested NO.
    """

    outcome = clean_value(row.get("D4_observable_outcome", "")).lower()

    if outcome in {
        "positive_reaction",
        "negative_reaction",
        "neutral_reaction",
    }:
        return "YES"

    return "NO"


def suggest_dm_redirect(row):
    """
    Existing automated D3 signal is used only as a pre-label suggestion.
    """

    d3 = clean_value(row.get("D3_dm_redirect", ""))

    try:
        score = float(d3)
    except ValueError:
        score = 0.0

    return "YES" if score >= 0.50 else "NO"


def suggest_evidence_complete(row):
    """
    Conservative automated suggestion.

    Evidence is suggested complete when:
      - there is a meaningful action, AND
      - there is either an observable outcome or reasonably specific
        support evidence.

    This remains a suggestion only.
    """

    action = suggest_action_observed(row)
    reaction = suggest_customer_reaction(row)
    dm = suggest_dm_redirect(row)

    d1 = clean_value(row.get("D1_context_specificity", ""))

    try:
        context_score = float(d1)
    except ValueError:
        context_score = 0.0

    if action == "YES" and (
        reaction == "YES"
        or context_score >= 0.60
    ) and dm == "NO":
        return "YES"

    return "NO"


def build_suggestions(row):
    return {
        "evidence_complete": suggest_evidence_complete(row),
        "action_observed": suggest_action_observed(row),
        "customer_reaction_observed": suggest_customer_reaction(row),
        "dm_redirect": suggest_dm_redirect(row),
    }


# ============================================================
# CREATE 150-EXAMPLE VERIFICATION SET
# ============================================================

def create_verification_file():
    if not SOURCE_SAMPLE.exists():
        st.error(
            f"Source sample not found:\n\n{SOURCE_SAMPLE}"
        )
        st.stop()

    source = pd.read_csv(SOURCE_SAMPLE)

    if len(source) < TARGET_SIZE:
        st.error(
            f"The source sample contains only {len(source)} rows. "
            f"{TARGET_SIZE} are required."
        )
        st.stop()

    # Reproducible selection from the existing 200-example sample.
    selected = source.sample(
        n=TARGET_SIZE,
        random_state=2026,
    ).copy()

    # Reset human-label columns.
    for column in LABEL_COLUMNS:
        selected[column] = ""

    selected[NOTE_COLUMN] = ""

    # Store automatic suggestions separately.
    for index, row in selected.iterrows():
        suggestions = build_suggestions(row)

        selected.loc[index, "suggested_evidence_complete"] = suggestions[
            "evidence_complete"
        ]
        selected.loc[index, "suggested_action_observed"] = suggestions[
            "action_observed"
        ]
        selected.loc[index, "suggested_customer_reaction_observed"] = suggestions[
            "customer_reaction_observed"
        ]
        selected.loc[index, "suggested_dm_redirect"] = suggestions[
            "dm_redirect"
        ]

    selected.to_csv(
        VERIFICATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return selected


# ============================================================
# LOAD DATA
# ============================================================

if not VERIFICATION_FILE.exists():
    create_verification_file()

df = pd.read_csv(VERIFICATION_FILE)

# Ensure expected columns exist.
for column in LABEL_COLUMNS + [NOTE_COLUMN]:
    if column not in df.columns:
        df[column] = ""

# Normalize labels.
for column in LABEL_COLUMNS:
    df[column] = df[column].fillna("").astype(str).str.upper().replace(
        {
            "Y": "YES",
            "N": "NO",
        }
    )

df[NOTE_COLUMN] = df[NOTE_COLUMN].fillna("").astype(str)


# ============================================================
# SESSION STATE
# ============================================================

if "current_index" not in st.session_state:
    st.session_state.current_index = 0


# Keep index valid.
st.session_state.current_index = max(
    0,
    min(
        st.session_state.current_index,
        len(df) - 1,
    ),
)


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_dataframe():
    df.to_csv(
        VERIFICATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🧠 Hiver AI — Human Verification</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Fast human verification for the 150-example golden evaluation set"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PROGRESS
# ============================================================

completed_mask = (
    df[LABEL_COLUMNS]
    .apply(lambda row: all(normalize_label(x) in {"YES", "NO"} for x in row), axis=1)
)

completed_count = int(completed_mask.sum())
remaining_count = len(df) - completed_count
progress = completed_count / len(df) if len(df) else 0.0

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Golden-set size", len(df))

with metric2:
    st.metric("Human verified", completed_count)

with metric3:
    st.metric("Remaining", remaining_count)

st.progress(progress)

st.markdown(
    f'<div class="progress-text">'
    f"Progress: {completed_count}/{len(df)} verified "
    f"({progress * 100:.1f}%)"
    f"</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Navigation")

    if st.button("⬅ Previous", use_container_width=True):
        st.session_state.current_index = max(
            0,
            st.session_state.current_index - 1,
        )
        st.rerun()

    if st.button("Next ➡", use_container_width=True):
        st.session_state.current_index = min(
            len(df) - 1,
            st.session_state.current_index + 1,
        )
        st.rerun()

    st.divider()

    jump_to = st.number_input(
        "Go to example",
        min_value=1,
        max_value=len(df),
        value=st.session_state.current_index + 1,
        step=1,
    )

    if st.button("Open example", use_container_width=True):
        st.session_state.current_index = int(jump_to) - 1
        st.rerun()

    st.divider()

    st.header("Golden-set rules")

    st.markdown(
        """
        **Evidence complete**

        YES only when the conversation contains enough concrete
        information plus meaningful support action or useful
        observable outcome.

        **Action observed**

        YES when AmazonHelp performs or clearly states a concrete
        support action.

        **Customer reaction**

        YES only when the customer gives an observable reaction
        after the relevant agent action.

        Silence does **not** mean solved.

        **DM redirect**

        YES when AmazonHelp redirects to DM/private messaging
        before a concrete public action.

        DM-only responses are not evidence of resolution.

        When uncertain, prefer **NO**.
        """
    )

    st.divider()

    st.warning(
        "The yellow suggestion boxes are automatic pre-labels. "
        "Do NOT blindly accept them. Read the conversation and "
        "make the final human decision."
    )


# ============================================================
# CURRENT RECORD
# ============================================================

idx = st.session_state.current_index
row = df.iloc[idx]

st.subheader(
    f"Conversation {idx + 1} of {len(df)}"
)

conversation_id = clean_value(row.get("conversation_id", ""))
customer_id = clean_value(row.get("customer_id", ""))

meta1, meta2, meta3 = st.columns(3)

with meta1:
    st.caption("Conversation ID")
    st.code(conversation_id or "N/A")

with meta2:
    st.caption("Customer ID")
    st.code(customer_id or "N/A")

with meta3:
    st.caption("Turn count")
    st.code(clean_value(row.get("turn_count", "N/A")))


# ============================================================
# AUTOMATIC SUGGESTIONS
# ============================================================

suggestions = build_suggestions(row)

with st.expander("🤖 Automatic pre-label suggestions", expanded=True):

    st.markdown(
        '<div class="suggestion">'
        "<b>These are suggestions only.</b> "
        "Verify every one against the actual conversation."
        "</div>",
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Evidence",
            suggestions["evidence_complete"],
        )

    with s2:
        st.metric(
            "Action",
            suggestions["action_observed"],
        )

    with s3:
        st.metric(
            "Reaction",
            suggestions["customer_reaction_observed"],
        )

    with s4:
        st.metric(
            "DM redirect",
            suggestions["dm_redirect"],
        )


# ============================================================
# CUSTOMER QUERY
# ============================================================

st.subheader("Customer query")

customer_query = clean_value(row.get("customer_query", ""))

st.markdown(
    f"""
    <div class="conversation-box customer">
        <b>Customer</b><br><br>
        {customer_query}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FULL CONVERSATION
# ============================================================

st.subheader("Historical conversation")

customer_messages = split_messages(
    row.get("customer_messages", "")
)

agent_messages = split_messages(
    row.get("agent_messages", "")
)

if not customer_messages and not agent_messages:

    conversation_text = clean_value(
        row.get("conversation_text", "")
    )

    if conversation_text:
        st.text_area(
            "Conversation",
            conversation_text,
            height=300,
            disabled=True,
        )

else:

    max_turns = max(
        len(customer_messages),
        len(agent_messages),
    )

    for turn in range(max_turns):

        if turn < len(customer_messages):
            message = customer_messages[turn]

            st.markdown(
                f"""
                <div class="conversation-box customer">
                    <b>Customer</b><br><br>
                    {message}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if turn < len(agent_messages):
            message = agent_messages[turn]

            st.markdown(
                f"""
                <div class="conversation-box agent">
                    <b>AmazonHelp</b><br><br>
                    {message}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# AUTOMATED FEATURES
# ============================================================

with st.expander("Automated evidence features — reference only"):

    feature_cols = [
        "D1_context_specificity",
        "D2_action_specificity",
        "D3_dm_redirect",
        "D4_observable_outcome",
        "evidence_completeness_score",
        "evidence_tier",
    ]

    feature_data = {}

    for column in feature_cols:
        if column in df.columns:
            feature_data[column] = clean_value(row[column])

    if feature_data:
        st.table(
            pd.DataFrame(
                {
                    "Feature": list(feature_data.keys()),
                    "Automated value": list(feature_data.values()),
                }
            )
        )

    st.caption(
        "These automated values must not be copied blindly into the "
        "human verification labels."
    )


# ============================================================
# HUMAN VERIFICATION
# ============================================================

st.subheader("Human verification")

st.markdown(
    "**Read the conversation first. Then verify or correct each suggestion.**"
)


def radio_with_default(
    label,
    current_value,
    suggested_value,
    key,
):

    options = [
        "Select...",
        "YES",
        "NO",
    ]

    if current_value in {"YES", "NO"}:
        default_index = options.index(current_value)
    elif suggested_value in {"YES", "NO"}:
        default_index = options.index(suggested_value)
    else:
        default_index = 0

    return st.radio(
        label,
        options,
        index=default_index,
        horizontal=True,
        key=key,
    )


col1, col2 = st.columns(2)

with col1:

    evidence_complete = radio_with_default(
        "1. Is the evidence complete?",
        normalize_label(row["evidence_complete"]),
        suggestions["evidence_complete"],
        f"evidence_{idx}",
    )

    action_observed = radio_with_default(
        "2. Was a concrete support action observed?",
        normalize_label(row["action_observed"]),
        suggestions["action_observed"],
        f"action_{idx}",
    )

with col2:

    customer_reaction_observed = radio_with_default(
        "3. Was an observable customer reaction seen after the action?",
        normalize_label(row["customer_reaction_observed"]),
        suggestions["customer_reaction_observed"],
        f"reaction_{idx}",
    )

    dm_redirect = radio_with_default(
        "4. Was there a DM/private redirect before a concrete public action?",
        normalize_label(row["dm_redirect"]),
        suggestions["dm_redirect"],
        f"dm_{idx}",
    )


current_notes = clean_value(row.get(NOTE_COLUMN, ""))

notes = st.text_area(
    "Notes — optional",
    value=current_notes,
    placeholder=(
        "Example: specific issue but only DM redirect; "
        "refund action observed but no customer reaction; "
        "ambiguous evidence."
    ),
    height=90,
    key=f"notes_{idx}",
)


# ============================================================
# SAVE / SKIP
# ============================================================

st.divider()

save_col, skip_col = st.columns(2)

with save_col:

    if st.button(
        "✅ Verify & Save → Next",
        type="primary",
        use_container_width=True,
    ):

        labels = {
            "evidence_complete": evidence_complete,
            "action_observed": action_observed,
            "customer_reaction_observed": customer_reaction_observed,
            "dm_redirect": dm_redirect,
        }

        invalid = [
            name
            for name, value in labels.items()
            if value not in {"YES", "NO"}
        ]

        if invalid:

            st.error(
                "Please answer all four questions before saving."
            )

        else:

            for name, value in labels.items():
                df.loc[idx, name] = value

            df.loc[idx, NOTE_COLUMN] = notes

            save_dataframe()

            st.session_state.current_index = min(
                len(df) - 1,
                idx + 1,
            )

            st.success(
                f"Saved human verification for example {idx + 1}."
            )

            st.rerun()


with skip_col:

    if st.button(
        "⏭ Skip for now",
        use_container_width=True,
    ):

        st.session_state.current_index = min(
            len(df) - 1,
            idx + 1,
        )

        st.rerun()


# ============================================================
# COMPLETION
# ============================================================

if completed_count == len(df):

    st.divider()

    st.success(
        "🎉 All 150 conversations have been human verified!"
    )

    st.subheader("Final verification distribution")

    for column in LABEL_COLUMNS:

        counts = (
            df[column]
            .value_counts()
            .reindex(["YES", "NO"], fill_value=0)
        )

        st.write(
            f"**{column}** — "
            f"YES: {int(counts['YES'])} | "
            f"NO: {int(counts['NO'])}"
        )

    st.info(
        "Next step: run the annotation validator and then use this "
        "golden set for the independent evaluation pipeline."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hiver SDE Intern Take-Home — Evidence-Completeness-Aware Customer Support RAG"
)