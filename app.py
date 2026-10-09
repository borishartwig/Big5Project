"""Streamlit app: estimate a personality type from big5.joblib.

Run with:  streamlit run app.py
(big5.joblib must be in the same folder as this file.)
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).parent / "big5.joblib"

# Column name -> question text, in the order they are shown
QUESTIONS = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": 'I seldom feel blue ("blue" meaning "sad", "depressed").',
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "C4": "I make a mess of things.",
    "A4": "I sympathize with others' feelings.",
}

GENDER_OPTIONS = ["Male", "Female", "Other"]
HAND_OPTIONS = ["Right", "Left", "Both"]


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["pipeline"], bundle["label_encoder"]


def trained_value(pipeline, column, choice):
    """Return the category spelling the model was trained with (ignores case),
    so a capitalisation difference can't silently turn the input into 'unknown'."""
    try:
        pre = pipeline.named_steps["preprocessor"]
        for _, encoder, cols in pre.transformers_:
            if column in list(cols):
                categories = list(encoder.categories_[list(cols).index(column)])
                for cat in categories:
                    if str(cat).lower() == choice.lower():
                        return cat
    except Exception:
        pass
    return choice


st.set_page_config(page_title="Personality type", page_icon="🧭", layout="centered")
st.title("Personality type")
st.write(
    "Rate how well each statement describes you, from "
    "**1 (no way)** to **5 (most definitely)**."
)

pipeline, label_encoder = load_model()

with st.form("questionnaire"):
    answers = {}
    for col, text in QUESTIONS.items():
        answers[col] = st.radio(
            f"**{col}**  {text}",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=col,
        )

    st.divider()
    age = st.number_input("**age**  Age in years", min_value=1, value=30, step=1)
    gender = st.radio("**gender**", GENDER_OPTIONS, index=2, horizontal=True)
    hand = st.radio("**hand**", HAND_OPTIONS, index=0, horizontal=True)

    submitted = st.form_submit_button("Evaluate")

if submitted:
    row = {**answers, "age": age,
           "gender": trained_value(pipeline, "gender", gender),
           "hand": trained_value(pipeline, "hand", hand)}
    X_new = pd.DataFrame([row])
    prediction = pipeline.predict(X_new)
    personality = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Your personality type is likely to be **{personality}**.")

st.caption(
    "An automated personality test is only an educated guess and can not be "
    "seen as a substitute for a talk with a professional."
)
