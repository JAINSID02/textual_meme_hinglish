# meme_llm/app/app.py
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg
from eval.predict import MemePredictor

st.set_page_config(page_title="Hinglish Meme Classifier", page_icon="😂", layout="centered")

CKPT = Path(cfg.paths.cls_dir) / "model_best.pt"
MIN_CONF = 0.40          # below this, show "not sure" instead of forcing a label

EXAMPLES = [
    "bhai kal exam hai aur maine syllabus khola bhi nahi 😭",
    "jab crush ne seen karke reply nahi diya",
    "boss: ek chhota sa kaam hai. kaam: poora weekend",
    "netaji ka vaada: 15 lakh. reality: 15 minute ka bhashan",
]


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return MemePredictor(CKPT)


st.title("Hinglish Meme Classifier")
st.caption("A small transformer, pretrained from scratch on Hinglish text and fine-tuned to sort memes into categories.")

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Couldn't load the classifier checkpoint at `{CKPT}`.\n\n{e}")
    st.stop()

if "text" not in st.session_state:
    st.session_state.text = ""

st.write("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.text = ex

text = st.text_area("Paste meme text (Hinglish)", key="text", height=110)

if st.button("Classify", type="primary", disabled=not text.strip()):
    result = predictor.predict(text, top_k=len(predictor.categories), min_confidence=MIN_CONF)

    if result["uncertain"]:
        st.warning(f"Not sure. Best guess: **{result['label']}** ({result['confidence']:.0%})")
    else:
        st.success(f"**{result['label']}**  ({result['confidence']:.0%} confidence)")

    chart = pd.DataFrame(result["top"], columns=["category", "probability"]).set_index("category")
    st.bar_chart(chart)

st.divider()
st.caption("Predictions come from the model's learned patterns and can be wrong, especially for sarcasm "
           "and memes that depend on an image. Confidence scores are rough guides, not calibrated probabilities.")