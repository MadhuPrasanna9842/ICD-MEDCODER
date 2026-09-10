import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import spacy
from pypdf import PdfReader
import io
import re
import json
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="MediCode AI — Clinical Coding & RCM Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Light Blue Clinical CSS Theme ---
st.markdown("""
    <style>
    /* Full Page & App Background */
    .stApp {
        background-color: #EBF4FC !important;
        color: #0F172A !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E0F2FE 0%, #BAE6FD 100%) !important;
        border-right: 1.5px solid #93C5FD !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    /* Ensure All Text & Labels are Crisp Dark */
    p, span, label, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
    }
    .stTextArea label, .stRadio label, .stTextInput label, .stFileUploader label {
        font-weight: 700 !important;
        color: #0369A1 !important;
        font-size: 14px !important;
    }
    
    /* Input Fields (Fixing Dark Mode Inversion) */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #93C5FD !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.06) !important;
    }
    .stTextArea textarea:focus {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2) !important;
    }

    /* Top Main Header Card */
    .header-box {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #075985 100%);
        padding: 20px 24px;
        border-radius: 14px;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.22);
        margin-bottom: 20px;
    }
    .header-box h2, .header-box p {
        color: #FFFFFF !important;
        margin: 0 !important;
    }
    
    /* Custom Result Cards */
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .clinical-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(2, 132, 199, 0.15);
    }
    
    .card-primary {
        border-left: 6px solid #059669;
        background: #F0FDF4;
    }
    .card-secondary {
        border-left: 6px solid #0284C7;
        background: #F0F9FF;
    }
    .card-cpt {
        border-left: 6px solid #6366F1;
        background: #EEF2FF;
    }
    .card-negated {
        border-left: 6px solid #DC2626;
        background: #FEF2F2;
    }

    /* Metric Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1.5px solid #BAE6FD;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.08);
    }
    .stat-val { font-size: 26px; font-weight: 800; color: #0284C7; }
    .stat-lbl { font-size: 11.5px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-top: 4px; }

    /* Highlights */
    .hl-active { background-color: #BBF7D0; color: #166534 !important; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
    .hl-cpt { background-color: #C7D2FE; color: #3730A3 !important; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
    .hl-neg { background-color: #FECACA; color: #991B1B !important; text-decoration: line-through; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- Procedural CPT Knowledge Base ---
CPT_PROCEDURE_REGISTRY = [
    {"cpt_code": "99214", "description": "Office/outpatient visit for evaluation & management, moderate complexity (30-39 mins)", "keywords": ["visit", "examination", "consultation", "follow up", "evaluation"], "rvu_cost": 135.00, "prior_auth": False},
    {"cpt_code": "93000", "description": "Electrocardiogram (ECG/EKG), routine 12-lead with interpretation & report", "keywords": ["ecg", "ekg", "electrocardiogram", "cardiac monitoring", "rhythm strip"], "rvu_cost": 48.00, "prior_auth": False},
    {"cpt_code": "71045", "description": "Radiologic examination, chest; single view", "keywords": ["chest x-ray", "cxr", "radiograph chest", "xray chest"], "rvu_cost": 68.00, "prior_auth": False},
    {"cpt_code": "31622", "description": "Diagnostic bronchoscopy with or without cell washing", "keywords": ["bronchoscopy", "airway inspection", "endobronchial exam"], "rvu_cost": 440.00, "prior_auth": True},
    {"cpt_code": "43239", "description": "Esophagogastroduodenoscopy (EGD) biopsy, single or multiple", "keywords": ["endoscopy", "upper gi endoscopy", "gastroscopy", "biopsy"], "rvu_cost": 395.00, "prior_auth": True},
    {"cpt_code": "94010", "description": "Spirometry with forced expiratory vital capacity (PFT)", "keywords": ["spirometry", "pulmonary function test", "pft", "lung function"], "rvu_cost": 90.00, "prior_auth": False},
    {"cpt_code": "80053", "description": "Comprehensive Metabolic Panel (CMP blood work)", "keywords": ["metabolic panel", "blood work", "cmp", "liver function test", "electrolytes"], "rvu_cost": 38.00, "prior_auth": False},
    {"cpt_code": "96372", "description": "Therapeutic/prophylactic injection; subcutaneous or intramuscular", "keywords": ["injection", "im injection", "administered medication", "intramuscular"], "rvu_cost": 55.00, "prior_auth": False}
]

# --- Database & Embeddings ---
@st.cache_resource
def load_system():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv("master_icd10_registry.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    
    code_col = next((c for c in ["icd10_code", "code", "icd_code", "icd10"] if c in df.columns), df.columns[0])
    desc_col = next((c for c in ["full_description", "description", "long_description", "desc"] if c in df.columns), df.columns[1])
    cat_col = next((c for c in ["category", "disease_category", "class"] if c in df.columns), None)
    ch_col = next((c for c in ["chapter", "icd_chapter"] if c in df.columns), None)
    
    client = chromadb.Client()
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    icd_collection = client.get_or_create_collection(name="master_icd10_blue_v1", embedding_function=emb_fn)
    if icd_collection.count() == 0:
        ids = df[code_col].astype(str).tolist()
        docs = df[desc_col].astype(str).tolist()
        metas = [{"category": str(df[cat_col].iloc[i]) if cat_col else "General",
                  "chapter": str(df[ch_col].iloc[i]) if ch_col else "General"} for i in range(len(df))]
        icd_collection.add(ids=ids, documents=docs, metadatas=metas)
        
    cpt_collection = client.get_or_create_collection(name="master_cpt_blue_v1", embedding_function=emb_fn)
    if cpt_collection.count() == 0:
        cpt_df = pd.DataFrame(CPT_PROCEDURE_REGISTRY)
        cpt_collection.add(
            ids=cpt_df["cpt_code"].tolist(),
            documents=cpt_df["description"].tolist(),
            metadatas=[{"keywords": " ".join(k), "cost": float(c)} for k, c in zip(cpt_df["keywords"], cpt_df["rvu_cost"])]
        )
        
    return nlp, icd_collection, cpt_collection, df, code_col, desc_col

nlp, icd_collection, cpt_collection, icd_df, code_col_name, desc_col_name = load_system()

# --- Linguistic Preprocessor ---
NON_CLINICAL_STOPWORDS = {
    "male", "female", "patient", "year-old", "man", "woman", "history", "day", "days", 
    "week", "weeks", "month", "months", "year", "years", "doctor", "hospital", "clinic", 
    "morning", "night", "today", "yesterday", "presents", "examination", "review", "complains", 
    "reports", "denies", "occasional", "persistent", "old", "presents with", "underwent", "performed"
}
NEGATION_TRIGGERS = ["no", "not", "denies", "without", "absent", "negative for", "ruled out", "free of"]

def clean_entity_text(phrase):
    phrase = re.sub(r"\b\d+[- ]*(year|yr)[- ]*old\b", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"[^a-zA-Z\s]", " ", phrase)
    words = [w.lower() for w in phrase.split() if w.lower() not in NON_CLINICAL_STOPWORDS and len(w) > 2]
    return " ".join(words).strip()

def parse_clinical_doc(text):
    doc = nlp(text)
    pos_findings = []
    neg_findings = []
    procedures = []
    
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        has_neg = any(re.search(rf"\b{neg}\b", sent_lower) for neg in NEGATION_TRIGGERS)
        
        for proc in CPT_PROCEDURE_REGISTRY:
            if any(re.search(rf"\b{kw}\b", sent_lower) for kw in proc["keywords"]):
                procedures.append(proc["cpt_code"])
                
        for chunk in sent.noun_chunks:
            cleaned = clean_entity_text(chunk.text)
            if cleaned and len(cleaned) > 2:
                if has_neg:
                    neg_findings.append(cleaned)
                else:
                    pos_findings.append(cleaned)
                    
    active_findings = [e for e in dict.fromkeys(pos_findings) if e not in neg_findings]
    neg_findings = list(dict.fromkeys(neg_findings))
    procedures = list(dict.fromkeys(procedures))
    return active_findings, neg_findings, procedures

def generate_highlighted_text(text, active_terms, neg_terms, procedures):
    annotated = text
    for neg in neg_terms:
        pattern = re.compile(re.escape(neg), re.IGNORECASE)
        annotated = pattern.sub(f'<span class="hl-neg">❌ {neg}</span>', annotated)
    for act in active_terms:
        pattern = re.compile(re.escape(act), re.IGNORECASE)
        annotated = pattern.sub(f'<span class="hl-active">🟢 {act}</span>', annotated)
    for proc_id in procedures:
        proc_obj = next((p for p in CPT_PROCEDURE_REGISTRY if p["cpt_code"] == proc_id), None)
        if proc_obj:
            for kw in proc_obj["keywords"]:
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                annotated = pattern.sub(f'<span class="hl-cpt">🔧 {kw}</span>', annotated)
    return annotated

def calculate_comorbidity_index(mapped_icd):
    score = 0
    weights = {"I10": 1, "E11": 1, "J44": 1, "K21": 1, "I25": 1, "I20": 1, "I50": 2, "N18": 2, "C34": 6, "I63": 1}
    for item in mapped_icd:
        prefix = item["ICD-10 Code"].split(".")[0]
        if prefix in weights:
            score += weights[prefix]
    if score == 0:
        return score, "Low Risk Tier", "98% (High Stability)", "#059669"
    elif score <= 2:
        return score, "Moderate Risk Tier", "90% (Standard Monitoring)", "#0284C7"
    else:
        return score, "High Risk / Complex Case", "72% (Intensive Care Plan Required)", "#DC2626"

def generate_fhir_bundle(patient_id, mapped_icd, mapped_cpt):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry": []
    }
    for i, icd in enumerate(mapped_icd):
        bundle["entry"].append({
            "fullUrl": f"urn:uuid:condition-{i+1}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis", "display": icd["Type"]}]}],
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd["ICD-10 Code"], "display": icd["Description"]}], "text": icd["Extracted Term"]},
                "subject": {"reference": f"Patient/{patient_id}"}
            }
        })
    for j, cpt in enumerate(mapped_cpt):
        bundle["entry"].append({
            "fullUrl": f"urn:uuid:procedure-{j+1}",
            "resource": {
                "resourceType": "Procedure",
                "status": "completed",
                "code": {"coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": cpt["CPT Code"], "display": cpt["Description"]}]},
                "subject": {"reference": f"Patient/{patient_id}"}
            }
        })
    return bundle

# --- SIDEBAR NAVIGATION (With Icons) ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="font-size: 38px;">🩺</div>
            <h3 style="margin: 0; color: #0369A1; font-weight: 800;">MediCode AI</h3>
            <span style="font-size: 11.5px; color: #475569; font-weight: 600;">Clinical Intelligence Suite</span>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    selected_page = st.radio(
        "NAVIGATION MENU",
        [
            "⚡ Clinical Coder",
            "🔍 Visual Highlighter",
            "📊 RCM & Risk Tier",
            "📑 FHIR R4 & Superbill",
            "🔎 Master Registry"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("""
        <div style="background: rgba(255,255,255,0.7); border: 1px solid #BAE6FD; border-radius: 8px; padding: 10px; font-size: 12px; color: #334155;">
            <b>System Status:</b> <span style="color: #059669; font-weight: bold;">● Online</span><br/>
            <b>Model:</b> SentenceTransformer<br/>
            <b>Negation Gate:</b> Active
        </div>
    """, unsafe_allow_html=True)

# --- Top Header Banner ---
st.markdown("""
<div class="header-box">
    <h2>🩺 MediCode AI — Autonomous Medical Coder & RCM Engine</h2>
    <p style="font-size: 13.5px; margin-top: 4px; opacity: 0.95;">Dense Semantic Vector Retrieval • Negation Boundary Guard • Dual ICD-10/CPT Coding • FHIR R4 Interoperability</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PAGE 1: CLINICAL CODER
# ==========================================
if selected_page == "⚡ Clinical Coder":
    col_in, col_out = st.columns([1.05, 0.95], gap="large")
    
    with col_in:
        st.markdown("#### 📝 Clinical Source Documentation")
        mode = st.radio("Select Ingestion Mode:", ["Preset Medical Note", "Custom Physician Text", "Upload PDF Document"], horizontal=True)
        
        note_content = ""
        if mode == "Preset Medical Note":
            preset_note = "64-year-old male presents with acute retrosternal burning chest pain and gastroesophageal reflux for 3 weeks. Patient underwent 12-lead electrocardiogram (ECG) and diagnostic upper GI endoscopy biopsy. Reports occasional dry cough. Patient denies fever, syncope, or hemoptysis."
            note_content = st.text_area("Physician Notes:", value=preset_note, height=180)
        elif mode == "Custom Physician Text":
            note_content = st.text_area("Physician Notes:", placeholder="Type or paste medical record here...", height=180)
        else:
            pdf_file = st.file_uploader("Upload Medical PDF", type=["pdf", "txt"])
            if pdf_file:
                if pdf_file.name.endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(pdf_file.read()))
                    for p in reader.pages:
                        note_content += p.extract_text() or ""
                else:
                    note_content = pdf_file.read().decode("utf-8")
                st.text_area("Extracted PDF Content:", value=note_content[:600] + "...", height=140, disabled=True)
                
        run_pipeline = st.button("🚀 Run Autonomous Medical Coding Pipeline", type="primary", use_container_width=True)
        
    with col_out:
        st.markdown("#### 🎯 Diagnostic & Procedural Codes")
        
        if run_pipeline and note_content.strip():
            with st.spinner("Executing linguistic parsing and semantic vector inference..."):
                active_terms, neg_terms, procedures = parse_clinical_doc(note_content)
                
                icd_records = []
                seen_icd = set()
                counter = 0
                
                for term in active_terms:
                    res = icd_collection.query(query_texts=[term], n_results=1)
                    if res["ids"] and len(res["ids"][0]) > 0:
                        code = res["ids"][0][0]
                        desc = res["documents"][0][0]
                        cat = res["metadatas"][0][0].get("category", "General")
                        ch = res["metadatas"][0][0].get("chapter", "General")
                        dist = res["distances"][0][0]
                        conf = max(0, min(100, int((1 - (dist / 2)) * 100)))
                        
                        if code not in seen_icd:
                            seen_icd.add(code)
                            diag_label = "Primary Diagnosis" if counter == 0 else f"Secondary Diagnosis #{counter}"
                            counter += 1
                            reimb = 280.00 if counter == 1 else 125.00
                            
                            icd_records.append({
                                "Type": diag_label,
                                "Extracted Term": term,
                                "ICD-10 Code": code,
                                "Description": desc,
                                "Category": cat,
                                "Chapter": ch,
                                "Confidence": f"{conf}%",
                                "Estimated Reimbursement ($)": reimb
                            })
                            
                cpt_records = []
                for cid in procedures:
                    p_obj = next(p for p in CPT_PROCEDURE_REGISTRY if p["cpt_code"] == cid)
                    cpt_records.append({
                        "Type": "Procedural CPT",
                        "CPT Code": p_obj["cpt_code"],
                        "Description": p_obj["description"],
                        "Estimated Fee ($)": p_obj["rvu_cost"],
                        "Prior Auth Required": p_obj["prior_auth"]
                    })
                    
                st.session_state["icd_records"] = icd_records
                st.session_state["cpt_records"] = cpt_records
                st.session_state["neg_terms"] = neg_terms
                st.session_state["active_terms"] = active_terms
                st.session_state["procedures"] = procedures
                st.session_state["raw_text"] = note_content
                
        if "icd_records" in st.session_state and st.session_state["icd_records"]:
            for i, icd in enumerate(st.session_state["icd_records"]):
                badge_class = "card-primary" if i == 0 else "card-secondary"
                st.markdown(f"""
                <div class="clinical-card {badge_class}">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 800; color: #0284C7;">
                        <span>● {icd['Type']}</span>
                        <span>Confidence: {icd['Confidence']}</span>
                    </div>
                    <div style="font-size: 15.5px; font-weight: 800; color: #0F172A; margin: 4px 0;">🏷️ {icd['ICD-10 Code']} — {icd['Description']}</div>
                    <div style="font-size: 12px; color: #475569;">Matched Term: <b>"{icd['Extracted Term']}"</b> | Est. Base DRG: <b>${icd['Estimated Reimbursement ($)']:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
            if st.session_state["cpt_records"]:
                st.markdown("<h5 style='color:#0369A1; margin-top:12px;'>🔧 Billable Procedural CPT Codes</h5>", unsafe_allow_html=True)
                for cpt in st.session_state["cpt_records"]:
                    pa_text = '<span style="color: #DC2626; font-weight: 700;">⚠️ Prior-Auth Required</span>' if cpt["Prior Auth Required"] else '<span style="color: #059669; font-weight: 700;">✅ Direct Claim</span>'
                    st.markdown(f"""
                    <div class="clinical-card card-cpt">
                        <div style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span style="font-weight: 700; color: #4F46E5;">PROCEDURE CODE</span>
                            <span>{pa_text}</span>
                        </div>
                        <div style="font-size: 14.5px; font-weight: 800; color: #1E1B4B; margin-top: 3px;">📌 CPT {cpt['CPT Code']} — {cpt['Description']}</div>
                        <div style="font-size: 12px; color: #4338CA; margin-top: 2px;">Standard RVU Allowed: <b>${cpt['Estimated Fee ($)']:.2f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if st.session_state["neg_terms"]:
                st.markdown("<h5 style='color:#DC2626; margin-top:12px;'>🚫 Ruled-Out Non-Billable Entities</h5>", unsafe_allow_html=True)
                for neg in st.session_state["neg_terms"]:
                    st.markdown(f'<div class="clinical-card card-negated" style="padding: 8px 12px; font-size: 12.5px;">❌ <b>Ruled Out:</b> <i>"{neg}"</i> (Excluded from ICD-10 Billing)</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: VISUAL HIGHLIGHTER
# ==========================================
elif selected_page == "🔍 Visual Highlighter":
    st.markdown("#### 🔍 Interactive Clinical Entity Highlighter")
    st.markdown("Visual verification of extracted active diagnoses, procedural cues, and negated symptoms:")
    
    if "raw_text" in st.session_state:
        highlighted = generate_highlighted_text(
            st.session_state["raw_text"],
            st.session_state["active_terms"],
            st.session_state["neg_terms"],
            st.session_state["procedures"]
        )
        
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #BAE6FD; border-radius: 12px; padding: 22px; line-height: 2.2; font-size: 15.5px; color: #1E293B; box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);">
            {highlighted}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div style="text-align: center;"><span class="hl-active">🟢 Active Diagnosis</span></div>', unsafe_allow_html=True)
        c2.markdown('<div style="text-align: center;"><span class="hl-cpt">🔧 CPT Procedure</span></div>', unsafe_allow_html=True)
        c3.markdown('<div style="text-align: center;"><span class="hl-neg">❌ Negated / Ruled Out</span></div>', unsafe_allow_html=True)
    else:
        st.info("Please execute the coding pipeline in '⚡ Clinical Coder' to see visual annotations.")

# ==========================================
# PAGE 3: RCM & RISK TIER
# ==========================================
elif selected_page == "📊 RCM & Risk Tier":
    st.markdown("#### 📊 Revenue Cycle Management & Clinical Comorbidity Risk")
    
    if "icd_records" in st.session_state and st.session_state["icd_records"]:
        tot_icd = sum(r["Estimated Reimbursement ($)"] for r in st.session_state["icd_records"])
        tot_cpt = sum(c["Estimated Fee ($)"] for c in st.session_state["cpt_records"])
        tot_val = tot_icd + tot_cpt
        
        cci, tier, surv, color = calculate_comorbidity_index(st.session_state["icd_records"])
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">${tot_val:.2f}</div><div class="stat-lbl">Total Claim Value</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state["icd_records"])}</div><div class="stat-lbl">Active ICD Codes</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{len(st.session_state["cpt_records"])}</div><div class="stat-lbl">CPT Procedures</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="stat-card"><div class="stat-val" style="color: {color};">CCI {cci}</div><div class="stat-lbl">{tier}</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1.5px solid #BAE6FD; border-radius: 12px; padding: 18px; margin-top: 10px; box-shadow: 0 2px 6px rgba(2, 132, 199, 0.08);">
            <h5 style="color: #0369A1; margin-bottom: 6px; font-weight: 800;">🩺 Charlson Comorbidity Prognosis & Severity</h5>
            <p style="font-size: 14px; color: #334155; margin-bottom: 4px;"><b>Clinical Severity Status:</b> <span style="color: {color}; font-weight: bold;">{tier} (Score: {cci})</span></p>
            <p style="font-size: 14px; color: #334155; margin-bottom: 0;"><b>Estimated 10-Year Clinical Survival Baseline:</b> {surv}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run clinical notes through the coder to unlock RCM metrics.")

# ==========================================
# PAGE 4: FHIR R4 & SUPERBILL
# ==========================================
elif selected_page == "📑 FHIR R4 & Superbill":
    st.markdown("#### 📑 Standardized HL7 FHIR R4 Interoperability & Billing Export")
    
    if "icd_records" in st.session_state and st.session_state["icd_records"]:
        pid = "PATIENT-9842"
        fhir_bundle = generate_fhir_bundle(pid, st.session_state["icd_records"], st.session_state["cpt_records"])
        json_txt = json.dumps(fhir_bundle, indent=2)
        
        f1, f2 = st.columns(2)
        with f1:
            st.download_button(
                label="📥 Download HL7 FHIR R4 JSON Bundle",
                data=json_txt,
                file_name=f"FHIR_R4_{pid}.json",
                mime="application/json",
                use_container_width=True
            )
        with f2:
            export_df = pd.DataFrame(st.session_state["icd_records"])
            st.download_button(
                label="📥 Export Certified Billing CSV",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"Billing_Record_{pid}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.json(fhir_bundle)
    else:
        st.info("Ingest notes in '⚡ Clinical Coder' to generate FHIR R4 bundles.")

# ==========================================
# PAGE 5: MASTER REGISTRY
# ==========================================
elif selected_page == "🔎 Master Registry":
    st.markdown("#### 🔎 Master ICD-10 & CPT Code Registry Explorer")
    query = st.text_input("Search Clinical Terms / Procedures (e.g. 'chest pain', 'gastric reflux', 'bronchoscopy'):")
    
    if query.strip():
        search_res = icd_collection.query(query_texts=[query], n_results=5)
        st.markdown(f"**Top Semantic ICD-10 Matches for:** *'{query}'*")
        
        for i in range(len(search_res["ids"][0])):
            cid = search_res["ids"][0][i]
            cdesc = search_res["documents"][0][i]
            ccat = search_res["metadatas"][0][i].get("category", "General")
            cch = search_res["metadatas"][0][i].get("chapter", "General")
            
            st.markdown(f"""
            <div class="clinical-card card-secondary">
                <div style="font-size: 15px; font-weight: 800; color: #0284C7;">🏷️ {cid} — {cdesc}</div>
                <div style="font-size: 12px; color: #64748B;">Category: {ccat} | Chapter: {cch}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.dataframe(icd_df, height=380, use_container_width=True)
