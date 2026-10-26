import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ollama
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FluAssist",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = MODEL_DIR / "febrile_random_forest_v2.pkl"
    feature_path = MODEL_DIR / "feature_columns_v2.pkl"
    config_path = MODEL_DIR / "model_config_v2.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    model = joblib.load(model_path)

    feature_columns = joblib.load(feature_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    return model, feature_columns, config


try:
    model, feature_columns, config = load_model()

    PROBABILITY_THRESHOLD = float(
        config["probability_threshold"]
    )

except Exception as e:

    st.error(
        f"Unable to load the FluAssist model: {e}"
    )

    st.stop()


# ============================================================
# SLM CONFIGURATION
# ============================================================

SLM_MODEL = "qwen2.5:3b-instruct"


# ============================================================
# FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "Fever": "Fever",
    "Headache": "Headache",
    "Nausea": "Nausea",
    "Muscle_Pain_Myalgia": "Muscle Pain / Myalgia",
    "Vomiting": "Vomiting",
    "Diarrhea": "Diarrhea",
    "Joint_Pain": "Joint Pain",
    "Arthritis": "Arthritis",
    "Joint_Swelling": "Joint Swelling",
    "Rash": "Rash",
    "Chills": "Chills",
    "Sweating": "Sweating",
    "Fatigue": "Fatigue",
    "Eye_Orbital_Pain": "Eye / Orbital Pain",
    "Conjunctivitis": "Conjunctivitis"
}


# ============================================================
# EXPLICIT NEGATION PATTERNS
# ============================================================

NEGATION_PATTERNS = {

    "Fever": [
        "no fever",
        "denies fever",
        "without fever",
        "fever absent",
        "no history of fever"
    ],

    "Headache": [
        "no headache",
        "denies headache",
        "without headache",
        "headache absent"
    ],

    "Nausea": [
        "no nausea",
        "denies nausea",
        "without nausea",
        "nausea absent"
    ],

    "Muscle_Pain_Myalgia": [
        "no body pain",
        "denies body pain",
        "without body pain",
        "no body ache",
        "denies body ache",
        "without body ache",
        "no muscle pain",
        "denies muscle pain",
        "without muscle pain",
        "no myalgia",
        "denies myalgia"
    ],

    "Vomiting": [
        "no vomiting",
        "denies vomiting",
        "without vomiting",
        "vomiting absent",
        "no vomiting is reported",
        "no episodes of vomiting",
        "no episode of vomiting",
        "no vomit",
        "denies vomit"
    ],

    "Diarrhea": [
        "no diarrhea",
        "denies diarrhea",
        "without diarrhea",
        "diarrhea absent",
        "no diarrhoea",
        "denies diarrhoea",
        "without diarrhoea"
    ],

    "Joint_Pain": [
        "no joint pain",
        "denies joint pain",
        "without joint pain",
        "joint pain absent",
        "no arthralgia",
        "denies arthralgia"
    ],

    "Arthritis": [
        "no arthritis",
        "denies arthritis",
        "without arthritis",
        "arthritis absent"
    ],

    "Joint_Swelling": [
        "no joint swelling",
        "denies joint swelling",
        "without joint swelling",
        "joint swelling absent",
        "no swollen joint",
        "denies swollen joint"
    ],

    "Rash": [
        "no rash",
        "denies rash",
        "without rash",
        "rash absent",
        "no skin rash"
    ],

    "Chills": [
        "no chills",
        "denies chills",
        "without chills",
        "chills absent",
        "no chill"
    ],

    "Sweating": [
        "no sweating",
        "denies sweating",
        "without sweating",
        "sweating absent"
    ],

    "Fatigue": [
        "no fatigue",
        "denies fatigue",
        "without fatigue",
        "fatigue absent",
        "not tired",
        "denies tiredness"
    ],

    "Eye_Orbital_Pain": [
        "no eye pain",
        "denies eye pain",
        "without eye pain",
        "eye pain absent",
        "no orbital pain",
        "denies orbital pain",
        "without orbital pain",
        "no retro-orbital pain",
        "denies retro-orbital pain"
    ],

    "Conjunctivitis": [
        "no conjunctivitis",
        "denies conjunctivitis",
        "without conjunctivitis",
        "conjunctivitis absent",
        "no red eyes",
        "denies red eyes",
        "without red eyes",
        "red eyes absent",
        "no red eye",
        "denies red eye"
    ]
}


# ============================================================
# POSITIVE FEATURE TERMS
# ============================================================

FEATURE_TERMS = {

    "Fever": [
        "fever",
        "febrile"
    ],

    "Headache": [
        "headache"
    ],

    "Nausea": [
        "nausea",
        "nauseated"
    ],

    "Muscle_Pain_Myalgia": [
        "body pain",
        "body ache",
        "muscle pain",
        "myalgia",
        "muscle ache"
    ],

    "Vomiting": [
        "vomiting",
        "vomit",
        "threw up",
        "throwing up"
    ],

    "Diarrhea": [
        "diarrhea",
        "diarrhoea",
        "loose stools"
    ],

    "Joint_Pain": [
        "joint pain",
        "arthralgia"
    ],

    "Arthritis": [
        "arthritis"
    ],

    "Joint_Swelling": [
        "joint swelling",
        "swollen joint"
    ],

    "Rash": [
        "rash",
        "skin rash"
    ],

    "Chills": [
        "chills",
        "chill"
    ],

    "Sweating": [
        "sweating",
        "sweats"
    ],

    "Fatigue": [
        "fatigue",
        "tiredness",
        "tired"
    ],

    "Eye_Orbital_Pain": [
        "eye pain",
        "orbital pain",
        "retro-orbital pain",
        "retroorbital pain"
    ],

    "Conjunctivitis": [
        "conjunctivitis",
        "red eyes",
        "red eye"
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

if "clinical_note" not in st.session_state:
    st.session_state.clinical_note = ""

if "extracted_features" not in st.session_state:
    st.session_state.extracted_features = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# NEW CASE / CLEAR
# ============================================================

def clear_case():

    st.session_state.clinical_note = ""
    st.session_state.extracted_features = None
    st.session_state.prediction_result = None


# ============================================================
# PAGE TITLE
# ============================================================

st.title("FluAssist")

st.subheader(
    "SLM-Based Clinical Decision Support System"
)

st.caption(
    "Prototype for clinician-assisted classification "
    "of Dengue, Malaria, Typhoid and Chikungunya."
)


# ============================================================
# NEW CASE BUTTON
# ============================================================

st.button(
    "New Case / Clear",
    on_click=clear_case
)


# ============================================================
# CLINICAL NOTE
# ============================================================

st.markdown("### 1. Clinical Note")

clinical_note = st.text_area(
    "Enter the doctor's clinical note:",
    height=180,
    placeholder=(
        "Example: Patient has fever for three days "
        "with severe headache, chills and vomiting."
    ),
    key="clinical_note"
)


# ============================================================
# SLM EXTRACTION FUNCTION
# ============================================================

def extract_features(note):

    prompt = f"""
You are a strict clinical information extraction system.

Your ONLY task is to extract clinical findings that are
explicitly stated in the clinical note.

DO NOT diagnose.
DO NOT predict a disease.
DO NOT infer symptoms from other symptoms.
DO NOT use general medical knowledge to fill missing information.

For every feature use exactly one of:

1 = explicitly present
0 = explicitly absent
null = not mentioned

Rules:

- If a feature is not mentioned, return null.
- Do NOT treat an unmentioned feature as absent.
- Do NOT treat an unmentioned feature as present.
- "No vomiting" means Vomiting = 0.
- "Denies vomiting" means Vomiting = 0.
- "Without vomiting" means Vomiting = 0.
- "No rash" means Rash = 0.
- "Denies joint pain" means Joint_Pain = 0.
- Vomiting does NOT imply Nausea.
- Nausea does NOT imply Vomiting.
- Body pain may map to Muscle_Pain_Myalgia.
- Body pain does NOT imply Joint_Pain.
- Fever does NOT imply Fatigue.

Examples:

Example 1:
Clinical note:
"Patient has fever and headache. No rash."

Correct:
Fever = 1
Headache = 1
Rash = 0
Nausea = null
Vomiting = null

Example 2:
Clinical note:
"Patient has fever and nausea. No vomiting."

Correct:
Fever = 1
Nausea = 1
Vomiting = 0

Example 3:
Clinical note:
"Patient has fever and headache."

Correct:
Fever = 1
Headache = 1
Nausea = null
Vomiting = null
Diarrhea = null
Rash = null

Features:

{chr(10).join(feature_columns)}

Clinical note:
<<<
{note}
>>>
"""

    # --------------------------------------------------------
    # JSON schema
    # --------------------------------------------------------

    feature_schema = {
        "type": "object",
        "properties": {
            feature: {
                "anyOf": [
                    {
                        "type": "integer",
                        "enum": [0, 1]
                    },
                    {
                        "type": "null"
                    }
                ]
            }
            for feature in feature_columns
        },
        "required": feature_columns
    }

    # --------------------------------------------------------
    # Call SLM
    # --------------------------------------------------------

    response = ollama.chat(
        model=SLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format=feature_schema,
        options={
            "temperature": 0
        }
    )

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    extracted = json.loads(
        response["message"]["content"]
    )

    # --------------------------------------------------------
    # Normalize note
    # --------------------------------------------------------

    note_lower = note.lower()

    # --------------------------------------------------------
    # Safety check 1:
    # Explicit negation always becomes 0
    # --------------------------------------------------------

    for feature, patterns in NEGATION_PATTERNS.items():

        if any(
            pattern in note_lower
            for pattern in patterns
        ):

            extracted[feature] = 0

    # --------------------------------------------------------
    # Safety check 2:
    # Explicit textual evidence becomes 1,
    # unless the feature is negated.
    #
    # If there is no textual evidence:
    #    -> null
    # --------------------------------------------------------

    for feature, terms in FEATURE_TERMS.items():

        has_textual_evidence = any(
            term in note_lower
            for term in terms
        )

        is_negated = any(
            pattern in note_lower
            for pattern in NEGATION_PATTERNS.get(
                feature,
                []
            )
        )

        # Explicitly negated
        if is_negated:

            extracted[feature] = 0

        # Explicitly mentioned and not negated
        elif has_textual_evidence:

            extracted[feature] = 1

        # Not mentioned
        else:

            extracted[feature] = None

    # --------------------------------------------------------
    # Ensure only expected features are returned
    # --------------------------------------------------------

    cleaned_features = {}

    for feature in feature_columns:

        value = extracted.get(feature)

        if value not in {0, 1, None}:

            value = None

        cleaned_features[feature] = value

    return cleaned_features


# ============================================================
# EXTRACT CLINICAL FEATURES
# ============================================================

if st.button(
    "Extract Clinical Features",
    type="primary"
):

    if not clinical_note.strip():

        st.warning(
            "Please enter a clinical note first."
        )

    else:

        with st.spinner(
            "Extracting clinical features using SLM..."
        ):

            try:

                extracted = extract_features(
                    clinical_note
                )

                st.session_state.extracted_features = (
                    extracted
                )

                st.session_state.prediction_result = None

            except Exception as e:

                st.error(
                    f"SLM extraction failed: {e}"
                )


# ============================================================
# DOCTOR REVIEW / EDIT
# ============================================================

if st.session_state.extracted_features is not None:

    st.markdown(
        "### 2. Review / Edit Extracted Features"
    )

    st.info(
        "Review the extracted symptoms and correct any "
        "value before prediction."
    )

    confirmed_features = {}

    for feature in feature_columns:

        current_value = (
            st.session_state
            .extracted_features
            .get(feature)
        )

        options = [
            "Not mentioned",
            "Present",
            "Absent"
        ]

        # Determine initial selection
        if current_value == 1:

            default_index = 1

        elif current_value == 0:

            default_index = 2

        else:

            default_index = 0

        selected = st.selectbox(
            FEATURE_LABELS.get(
                feature,
                feature
            ),
            options,
            index=default_index,
            key=f"confirm_{feature}"
        )

        if selected == "Present":

            confirmed_features[feature] = 1

        elif selected == "Absent":

            confirmed_features[feature] = 0

        else:

            confirmed_features[feature] = np.nan


    # ========================================================
    # CLINICAL CLASSIFICATION
    # ========================================================

    st.markdown(
        "### 3. Clinical Classification"
    )

    if st.button(
        "Confirm Features and Predict",
        type="primary"
    ):

        try:

            # -----------------------------------------------
            # Create feature vector
            # -----------------------------------------------

            input_df = pd.DataFrame(
                [
                    confirmed_features
                ],
                columns=feature_columns
            )

            # -----------------------------------------------
            # Random Forest prediction
            # -----------------------------------------------

            probabilities = model.predict_proba(
                input_df
            )[0]

            classes = (
                model
                .named_steps["classifier"]
                .classes_
            )

            max_index = int(
                np.argmax(probabilities)
            )

            max_probability = float(
                probabilities[max_index]
            )

            predicted_class = classes[max_index]

            # -----------------------------------------------
            # Probability-based rejection
            # -----------------------------------------------

            if max_probability >= PROBABILITY_THRESHOLD:

                final_result = predicted_class
                status = "Accepted"

            else:

                final_result = (
                    "Unknown / Uncertain Case"
                )

                status = "Rejected"

            # -----------------------------------------------
            # Store result
            # -----------------------------------------------

            st.session_state.prediction_result = {
                "prediction": str(predicted_class),
                "probabilities": {
                    str(
                        disease
                    ): float(
                        probability
                    )
                    for disease, probability
                    in zip(
                        classes,
                        probabilities
                    )
                },
                "maximum_probability": max_probability,
                "final_result": final_result,
                "status": status
            }

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# RESULT
# ============================================================

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    st.markdown(
        "### 4. Result"
    )

    if result["status"] == "Accepted":

        st.success(
            "Supported disease prediction: "
            f"{result['final_result']}"
        )

    else:

        st.warning(
            "Unknown / Uncertain Case"
        )

    st.write(
        "Maximum class probability: "
        f"{result['maximum_probability']:.4f}"
    )

    st.write(
        "Provisional threshold: "
        f"{PROBABILITY_THRESHOLD:.2f}"
    )

    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.markdown(
        "#### Class Probabilities"
    )

    probability_df = pd.DataFrame({
        "Disease": list(
            result["probabilities"].keys()
        ),
        "Probability": list(
            result["probabilities"].values()
        )
    })

    probability_df["Probability"] = (
        probability_df["Probability"]
        .round(4)
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        "#### Model Information"
    )

    st.write(
        f"Model: Random Forest V2 Balanced"
    )

    st.write(
        f"Features: {len(feature_columns)}"
    )

    st.write(
        "Supported diseases: "
        "Dengue, Malaria, Typhoid, Chikungunya"
    )

    st.write(
        f"Provisional rejection threshold: "
        f"{PROBABILITY_THRESHOLD:.2f}"
    )

    # ========================================================
    # CLINICAL DISCLAIMER
    # ========================================================

    st.info(
        "This prototype supports clinical decision-making "
        "and does not replace clinical judgment or "
        "confirmatory investigation."
    )