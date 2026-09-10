import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import spacy
from pypdf import PdfReader
import io
import re

st.set_page_config(
    page_title="AI Clinical ICD-10 Coding & Billing Platform",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #3B82F6; margin-bottom: 2px; }
    .sub-title { font-size: 14px; color: #9CA3AF; margin-bottom: 20px; }
    .entity-tag { display: inline-block; background-color: #1E3A8A; color: #DBEAFE; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; margin: 2px; }
    .primary-card { background: #064E3B; border: 1px solid #059669; border-left: 6px solid #10B981; border-radius: 8px; padding: 14px; margin-bottom: 12px; color: #FFFFFF; }
    .secondary-card { background: #1E293B; border: 1px solid #334155; border-left: 6px solid #3B82F6; border-radius: 8px; padding: 14px; margin-bottom: 12px; color: #FFFFFF; }
    .negated-card { background: #450A0A; border: 1px solid #7F1D1D; border-left: 6px solid #EF4444; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; color: #FECACA; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv("master_icd10_registry.csv")
    
    client = chromadb.PersistentClient(path="./chroma_db_store")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="master_icd10_persistent", embedding_function=emb_fn)
    return nlp, collection, df

nlp, collection, icd_df = load_system()

NON_CLINICAL_STOPWORDS = {
    "male", "female", "patient", "year-old", "man", "woman", "history", "day", "days", 
    "week", "weeks", "month", "months", "year", "years", "doctor", "dinner", "lunch", 
    "breakfast", "hospital", "clinic", "morning", "night", "today", "yesterday", "presents",
    "meals", "examination", "review", "complains", "reports", "denies", "occasional", "persistent",
    "year", "old", "presents with"
}

NEGATION_TRIGGERS = ["no", "not", "denies", "without", "absent", "negative for", "ruled out"]

def clean_entity_text(phrase):
    phrase = re.sub(r"\b\d+[- ]*(year|yr)[- ]*old\b", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"[^a-zA-Z\s]", " ", phrase)
    words = [w.lower() for w in phrase.split() if w.lower() not in NON_CLINICAL_STOPWORDS and len(w) > 2]
    return " ".join(words).strip()

def parse_clinical_statements(text):
    doc = nlp(text)
    pos_entities = []
    neg_entities = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        has_negation = any(re.search(rf"\b{neg}\b", sent_text) for neg in NEGATION_TRIGGERS)
        
        for chunk in sent.noun_chunks:
            cleaned = clean_entity_text(chunk.text)
            if cleaned and len(cleaned) > 2:
                if has_negation:
                    neg_entities.append(cleaned)
                else:
                    pos_entities.append(cleaned)
                    
    # Remove duplicate and cross-negated entities
    clean_pos = [e for e in dict.fromkeys(pos_entities) if e not in neg_entities]
    clean_neg = list(dict.fromkeys(neg_entities))
    return clean_pos, clean_neg

# Streamlit Interface Layout
st.markdown('<div class="main-title">🩺 Automated Clinical ICD-10 Coder & Billing System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic Retrieval Engine with Rigorous Entity Cleaning & Negation Filtering</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader("📝 Clinical Source Documentation")
    input_type = st.radio("Choose Input Mode:", ["Direct Text Input", "Upload Medical Document (.pdf / .txt)"], horizontal=True)
    
    extracted_text = ""
    if input_type == "Direct Text Input":
        default_text = "56-year-old male presents with persistent retrosternal burning sensation after meals and acid reflux. Patient denies fever or headache. Reports occasional dry cough."
        extracted_text = st.text_area("Clinical Note / Summary:", value=default_text, height=180)
    else:
        uploaded_file = st.file_uploader("Upload Patient Discharge Summary / Note", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text() or ""
            else:
                extracted_text = uploaded_file.read().decode("utf-8")
            st.text_area("Extracted Document Content:", value=extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""), height=150, disabled=True)
            
    analyze_btn = st.button("⚡ Run Clinical Code Mapping", type="primary")

with col2:
    st.subheader("🎯 Diagnostic Codes & Billing Classification")
    
    if analyze_btn and extracted_text.strip():
        pos_entities, neg_entities = parse_clinical_statements(extracted_text)
        
        if pos_entities:
            st.markdown("**Active Clinical Findings (Billable):**")
            st.markdown("".join([f'<span class="entity-tag">📌 {e}</span>' for e in pos_entities]), unsafe_allow_html=True)
            st.write("")
            
            seen_codes = set()
            table_records = []
            bill_idx = 0
            
            for entity in pos_entities:
                res = collection.query(query_texts=[entity], n_results=1)
                
                if res["ids"] and len(res["ids"][0]) > 0:
                    code = res["ids"][0][0]
                    desc = res["documents"][0][0]
                    category = res["metadatas"][0][0]["category"]
                    chapter = res["metadatas"][0][0]["chapter"]
                    dist = res["distances"][0][0]
                    confidence = max(0, min(100, int((1 - (dist / 2)) * 100)))
                    
                    if code not in seen_codes:
                        seen_codes.add(code)
                        diag_type = "Primary Diagnosis" if bill_idx == 0 else f"Secondary Diagnosis #{bill_idx}"
                        card_class = "primary-card" if bill_idx == 0 else "secondary-card"
                        bill_idx += 1
                        
                        table_records.append({
                            "Type": diag_type,
                            "Extracted Term": entity,
                            "ICD-10 Code": code,
                            "Description": desc,
                            "Category": category,
                            "Chapter": chapter,
                            "Confidence": f"{confidence}%"
                        })
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">● {diag_type}</span>
                                <span style="font-weight: 700; font-size: 13px;">Match: {confidence}%</span>
                            </div>
                            <div style="font-size: 16px; font-weight: bold; margin-top: 4px;">🏷️ {code} — {desc}</div>
                            <div style="font-size: 12px; opacity: 0.85; margin-top: 2px;">Matched: <i>"{entity}"</i> | Category: {category} | Chapter: {chapter}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if neg_entities:
                st.markdown("**Negated / Ruled-Out Conditions (Non-Billable):**")
                for neg_term in neg_entities:
                    st.markdown(f"""
                    <div class="negated-card">
                        🚫 <strong>Ruled Out:</strong> <i>"{neg_term}"</i> (Excluded from ICD-10 Billing)
                    </div>
                    """, unsafe_allow_html=True)

            if table_records:
                st.divider()
                result_df = pd.DataFrame(table_records)
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Clinical Billing Record (CSV)",
                    data=csv_data,
                    file_name="clinical_billing_icd10.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No active clinical entities detected. Refine the notes or symptoms.")
    elif not extracted_text.strip():
        st.warning("Please provide clinical documentation text or upload a document.")
    else:
        st.info("Click 'Run Clinical Code Mapping' to parse the clinical summary.")

st.divider()
with st.expander("📚 Explore Master ICD-10 Database"):
    st.dataframe(icd_df)
