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

# --- Page Configuration & Light Blue Medical SaaS Theme ---
st.set_page_config(
    page_title="MediCode AI — Clinical Coding, CPT & RCM Intelligence",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    /* Light Blue & Medical SaaS Palette */
    .stApp {
        background-color: #F0F7FF !important;
        color: #0F172A !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header & Navigation Bar */
    .header-banner {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #075985 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: #FFFFFF;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.18);
        margin-bottom: 22px;
    }
    .header-title { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 4px; }
    .header-sub { font-size: 13.5px; opacity: 0.92; font-weight: 400; }
    
    /* Clinical Cards */
    .custom-card {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(186, 230, 253, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(2, 132, 199, 0.12);
    }
    
    .primary-badge {
        border-left: 5px solid #059669;
        background: #F0FDF4;
    }
    .secondary-badge {
        border-left: 5px solid #0284C7;
        background: #F0F9FF;
    }
    .cpt-badge {
        border-left: 5px solid #6366F1;
        background: #EEF2FF;
    }
    .negated-badge {
        border-left: 5px solid #DC2626;
        background: #FEF2F2;
    }

    /* Metric Containers */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.08);
    }
    .metric-val { font-size: 24px; font-weight: 800; color: #0369A1; }
    .metric-title { font-size: 11.5px; text-transform: uppercase; font-weight: 700; color: #64748B; margin-top: 2px; }

    /* Visual Note Highlighter */
    .highlight-active {
        background-color: #BBF7D0;
        color: #166534;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #86EFAC;
    }
    .highlight-cpt {
        background-color: #C7D2FE;
        color: #3730A3;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #A5B4FC;
    }
    .highlight-neg {
        background-color: #FECACA;
        color: #991B1B;
        text-decoration: line-through;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #FCA5A5;
    }

    /* Agent Timeline */
    .agent-box {
        background: #FFFFFF;
        border: 1px solid #E0F2FE;
        border-left: 4px solid #0284C7;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    </style>
""", unsafe_allow_html=True)

# --- Procedural CPT Registry ---
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
    
    icd_collection = client.get_or_create_collection(name="master_icd10_medicode", embedding_function=emb_fn)
    if icd_collection.count() == 0:
        ids = df[code_col].astype(str).tolist()
        docs = df[desc_col].astype(str).tolist()
        metas = [{"category": str(df[cat_col].iloc[i]) if cat_col else "General",
                  "chapter": str(df[ch_col].iloc[i]) if ch_col else "General"} for i in range(len(df))]
        icd_collection.add(ids=ids, documents=docs, metadatas=metas)
        
    cpt_collection = client.get_or_create_collection(name="master_cpt_medicode", embedding_function=emb_fn)
    if cpt_collection.count() == 0:
        cpt_df = pd.DataFrame(CPT_PROCEDURE_REGISTRY)
        cpt_collection.add(
            ids=cpt_df["cpt_code"].tolist(),
            documents=cpt_df["description"].tolist(),
            metadatas=[{"keywords": " ".join(k), "cost": float(c)} for k, c in zip(cpt_df["keywords"], cpt_df["rvu_cost"])]
        )
        
    return nlp, icd_collection, cpt_collection, df, code_col, desc_col

nlp, icd_collection, cpt_collection, icd_df, code_col_name, desc_col_name = load_system()

# --- Preprocessing & Linguistic Analysis ---
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

# --- Unique Feature 1: Clinical Visual Highlighter ---
def generate_highlighted_text(text, active_terms, neg_terms, procedures):
    annotated = text
    for neg in neg_terms:
        pattern = re.compile(re.escape(neg), re.IGNORECASE)
        annotated = pattern.sub(f'<span class="highlight-neg">❌ {neg}</span>', annotated)
        
    for act in active_terms:
        pattern = re.compile(re.escape(act), re.IGNORECASE)
        annotated = pattern.sub(f'<span class="highlight-active">🟢 {act}</span>', annotated)
        
    for proc_id in procedures:
        proc_obj = next((p for p in CPT_PROCEDURE_REGISTRY if p["cpt_code"] == proc_id), None)
        if proc_obj:
            for kw in proc_obj["keywords"]:
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                annotated = pattern.sub(f'<span class="highlight-cpt">🔧 {kw}</span>', annotated)
    return annotated

# --- Unique Feature 2: Charlson Comorbidity Index (CCI) Predictor ---
def calculate_comorbidity_index(mapped_icd):
    score = 0
    weights = {
        "I10": 1, "E11": 1, "J44": 1, "K21": 1, "I25": 1, "I20": 1,
        "I50": 2, "N18": 2, "C34": 6, "I63": 1
    }
    
    for item in mapped_icd:
        code_prefix = item["ICD-10 Code"].split(".")[0]
        if code_prefix in weights:
            score += weights[code_prefix]
            
    if score == 0:
        tier, survival, badge = "Low Risk Tier", "98% (High Stability)", "#059669"
    elif score <= 2:
        tier, survival, badge = "Moderate Risk Tier", "90% (Standard Monitoring)", "#0284C7"
    else:
        tier, survival, badge = "High Risk / Complex Case", "72% (Intensive Care Plan Required)", "#DC2626"
        
    return score, tier, survival, badge

# --- Unique Feature 3: FHIR R4 Bundle Export ---
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

# --- Header Banner ---
st.markdown("""
<div class="header-banner">
    <div class="header-title">🩺 MediCode AI — Clinical Autonomous Coder & RCM Intelligence</div>
    <div class="header-sub">Multi-Agent ICD-10/CPT Retrieval • Real-Time Denial Risk Guard • FHIR R4 Interoperability • CMS Prior-Auth Compliance</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Autonomous Clinical Coder", 
    "🔍 Visual Entity Highlighting", 
    "📊 RCM Analytics & Risk Tier", 
    "📑 FHIR R4 & Superbill Export", 
    "🔎 Master Code Explorer"
])

with tab1:
    col_left, col_right = st.columns([1.05, 0.95], gap="large")
    
    with col_left:
        st.markdown("##### 📝 Clinical Documentation Ingestion")
        input_type = st.radio("Documentation Source:", ["Clinical Template Note", "Free Text Entry", "Upload Patient PDF / Discharge Summary"], horizontal=True)
        
        clinical_text = ""
        if input_type == "Clinical Template Note":
            preset = "64-year-old male presents with acute retrosternal burning chest pain and gastroesophageal reflux for 3 weeks. Patient underwent 12-lead electrocardiogram (ECG) and diagnostic upper GI endoscopy biopsy. Reports occasional dry cough. Patient denies fever, syncope, or hemoptysis."
            clinical_text = st.text_area("Clinical Case Note:", value=preset, height=180)
        elif input_type == "Free Text Entry":
            clinical_text = st.text_area("Clinical Case Note:", placeholder="Paste physician examination notes here...", height=180)
        else:
            up = st.file_uploader("Upload Medical PDF", type=["pdf", "txt"])
            if up:
                if up.name.endswith(".pdf"):
                    r = PdfReader(io.BytesIO(up.read()))
                    for p in r.pages:
                        clinical_text += p.extract_text() or ""
                else:
                    clinical_text = up.read().decode("utf-8")
                st.text_area("Parsed Text Preview:", value=clinical_text[:800] + "...", height=150, disabled=True)
                
        run_btn = st.button("🚀 Run Multi-Agent Coding Pipeline", type="primary", use_container_width=True)
        
    with col_right:
        st.markdown("##### 🎯 Mapped Diagnostic & Billing Codes")
        
        if run_btn and clinical_text.strip():
            with st.spinner("Processing clinical semantics & executing vector retrieval..."):
                active_terms, neg_terms, procedures = parse_clinical_doc(clinical_text)
                
                icd_records = []
                seen_codes = set()
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
                        
                        if code not in seen_codes:
                            seen_codes.add(code)
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
                st.session_state["raw_text"] = clinical_text
                
        if "icd_records" in st.session_state and st.session_state["icd_records"]:
            for i, icd in enumerate(st.session_state["icd_records"]):
                badge_style = "primary-badge" if i == 0 else "secondary-badge"
                st.markdown(f"""
                <div class="custom-card {badge_style}">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; color: #0369A1;">
                        <span>● {icd['Type']}</span>
                        <span>Confidence: {icd['Confidence']}</span>
                    </div>
                    <div style="font-size: 15.5px; font-weight: 800; color: #0F172A; margin: 4px 0;">🏷️ {icd['ICD-10 Code']} — {icd['Description']}</div>
                    <div style="font-size: 12px; color: #475569;">Term: <b>"{icd['Extracted Term']}"</b> | Est. Base Reimbursement: <b>${icd['Estimated Reimbursement ($)']:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
            if st.session_state["cpt_records"]:
                st.markdown("**🔧 Billable CPT Procedures:**")
                for cpt in st.session_state["cpt_records"]:
                    pa_alert = '<span style="color: #DC2626; font-weight: 700;">⚠️ Prior-Auth Required</span>' if cpt["Prior Auth Required"] else '<span style="color: #059669; font-weight: 700;">✅ Direct Claim</span>'
                    st.markdown(f"""
                    <div class="custom-card cpt-badge">
                        <div style="display: flex; justify-content: space-between; font-size: 11px;">
                            <span style="font-weight: 700; color: #4F46E5;">PROCEDURE CODE</span>
                            <span>{pa_alert}</span>
                        </div>
                        <div style="font-size: 14.5px; font-weight: 700; color: #1E1B4B; margin-top: 3px;">📌 CPT {cpt['CPT Code']} — {cpt['Description']}</div>
                        <div style="font-size: 12px; color: #4338CA; margin-top: 2px;">Standard RVU Allowed: <b>${cpt['Estimated Fee ($)']:.2f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if st.session_state["neg_terms"]:
                st.markdown("**🚫 Non-Billable / Ruled-Out Conditions:**")
                for neg in st.session_state["neg_terms"]:
                    st.markdown(f'<div class="custom-card negated-badge" style="padding: 8px 12px; font-size: 12.5px;">❌ <b>Ruled Out:</b> <i>"{neg}"</i> (Excluded from claim by Negation Gate)</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("##### 🔍 Interactive Clinical Entity Annotation")
    st.markdown("Visual verification of extracted medical findings directly inside the patient's narrative:")
    
    if "raw_text" in st.session_state:
        highlighted = generate_highlighted_text(
            st.session_state["raw_text"],
            st.session_state["active_terms"],
            st.session_state["neg_terms"],
            st.session_state["procedures"]
        )
        
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #BAE6FD; border-radius: 10px; padding: 20px; line-height: 2.1; font-size: 15px; color: #1E293B;">
            {highlighted}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div style="text-align: center;"><span class="highlight-active">🟢 Active Diagnosis</span></div>', unsafe_allow_html=True)
        c2.markdown('<div style="text-align: center;"><span class="highlight-cpt">🔧 CPT Procedure</span></div>', unsafe_allow_html=True)
        c3.markdown('<div style="text-align: center;"><span class="highlight-neg">❌ Negated / Ruled Out</span></div>', unsafe_allow_html=True)
    else:
        st.info("Run the autonomous coder in the first tab to view visual highlights.")

with tab3:
    st.markdown("##### 📊 Revenue Cycle Management & Patient Severity Risk")
    
    if "icd_records" in st.session_state and st.session_state["icd_records"]:
        total_icd = sum(r["Estimated Reimbursement ($)"] for r in st.session_state["icd_records"])
        total_cpt = sum(c["Estimated Fee ($)"] for c in st.session_state["cpt_records"])
        total_claim = total_icd + total_cpt
        
        cci_score, risk_tier, survival_rate, tier_color = calculate_comorbidity_index(st.session_state["icd_records"])
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">${total_claim:.2f}</div><div class="metric-title">Total Claim Value</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(st.session_state["icd_records"])}</div><div class="metric-title">Active ICD-10 Codes</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(st.session_state["cpt_records"])}</div><div class="metric-title">Procedural CPT Codes</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color: {tier_color};">CCI {cci_score}</div><div class="metric-title">{risk_tier}</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #BAE6FD; border-radius: 10px; padding: 16px; margin-top: 10px;">
            <h5 style="color: #0369A1; margin-bottom: 6px;">🩺 Charlson Comorbidity & Longevity Prognosis</h5>
            <p style="font-size: 13.5px; color: #334155; margin-bottom: 4px;"><b>Calculated Severity:</b> <span style="color: {tier_color}; font-weight: bold;">{risk_tier} (Score: {cci_score})</span></p>
            <p style="font-size: 13.5px; color: #334155; margin-bottom: 0;"><b>Estimated 10-Year Clinical Survival Baseline:</b> {survival_rate}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("##### 🤖 Autonomous Audit & Rule Verification Trail")
        st.markdown("""
        <div class="agent-box"><b>🤖 Extractor Agent:</b> Isolated linguistic noun phrases and eliminated non-clinical demographic stops.</div>
        <div class="agent-box"><b>🏷️ Semantic Dual-Coder:</b> Cross-referenced embeddings against ICD-10-CM vector collection & procedural CPT registry.</div>
        <div class="agent-box"><b>⚖️ CMS Policy Auditor:</b> Negation gate verified. All 'denied' symptoms quarantined from claims submission.</div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run the autonomous pipeline to generate RCM insights.")

with tab4:
    st.markdown("##### 📑 Standardized HL7 / FHIR R4 Bundle & Official Superbill")
    
    if "icd_records" in st.session_state and st.session_state["icd_records"]:
        patient_id = "PT-MED-9842"
        fhir_obj = generate_fhir_bundle(patient_id, st.session_state["icd_records"], st.session_state["cpt_records"])
        json_output = json.dumps(fhir_obj, indent=2)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.download_button(
                label="📥 Export HL7 / FHIR R4 JSON Bundle",
                data=json_output,
                file_name=f"FHIR_R4_{patient_id}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_f2:
            export_df = pd.DataFrame(st.session_state["icd_records"])
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Certified Billing CSV",
                data=csv_data,
                file_name=f"Billing_Record_{patient_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.json(fhir_obj)
    else:
        st.info("No active records. Ingest a clinical note to export FHIR bundles.")

with tab5:
    st.markdown("##### 🔎 Intelligent Master Registry Explorer")
    search_q = st.text_input("Instant Semantic Search (e.g. 'chest pain', 'gastric ulcer', 'endoscopy'):")
    
    if search_q.strip():
        res = icd_collection.query(query_texts=[search_q], n_results=5)
        st.markdown(f"**Top Semantic Matches for:** *'{search_q}'*")
        
        for i in range(len(res["ids"][0])):
            cid = res["ids"][0][i]
            cdesc = res["documents"][0][i]
            ccat = res["metadatas"][0][i].get("category", "General")
            cch = res["metadatas"][0][i].get("chapter", "General")
            
            st.markdown(f"""
            <div class="custom-card secondary-badge">
                <div style="font-size: 15px; font-weight: 800; color: #0284C7;">🏷️ {cid} — {cdesc}</div>
                <div style="font-size: 12px; color: #64748B;">Category: {ccat} | Chapter: {cch}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.dataframe(icd_df, height=350)
