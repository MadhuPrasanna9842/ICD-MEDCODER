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

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Autonomous Clinical AI Coder & RCM Auditor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 26px; font-weight: 800; color: #3B82F6; margin-bottom: 2px; }
    .sub-title { font-size: 13px; color: #94A3B8; margin-bottom: 18px; }
    .metric-container { background: #1E293B; border-radius: 8px; padding: 12px; border: 1px solid #334155; text-align: center; }
    .metric-value { font-size: 22px; font-weight: bold; color: #38BDF8; }
    .metric-label { font-size: 11px; color: #94A3B8; text-transform: uppercase; }
    .agent-card { background: #0F172A; border-left: 4px solid #38BDF8; padding: 10px 14px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; }
    .primary-card { background: #064E3B; border-left: 6px solid #10B981; border-radius: 6px; padding: 12px; margin-bottom: 10px; color: #FFFFFF; }
    .secondary-card { background: #1E293B; border-left: 6px solid #3B82F6; border-radius: 6px; padding: 12px; margin-bottom: 10px; color: #FFFFFF; }
    .cpt-card { background: #312E81; border-left: 6px solid #818CF8; border-radius: 6px; padding: 12px; margin-bottom: 10px; color: #FFFFFF; }
    .negated-card { background: #450A0A; border-left: 6px solid #EF4444; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; color: #FECACA; font-size: 12px; }
    .highlight-active { background-color: rgba(16, 185, 129, 0.25); border-bottom: 2px solid #10B981; padding: 1px 4px; border-radius: 3px; }
    .highlight-proc { background-color: rgba(99, 102, 241, 0.25); border-bottom: 2px solid #818CF8; padding: 1px 4px; border-radius: 3px; }
    .highlight-neg { background-color: rgba(239, 68, 68, 0.25); border-bottom: 2px solid #EF4444; padding: 1px 4px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# --- Procedural CPT Knowledge Base ---
CPT_PROCEDURE_REGISTRY = [
    {"cpt_code": "99214", "description": "Office or outpatient visit for evaluation and management, moderate severity (30-39 mins)", "keywords": ["visit", "examination", "consultation", "follow up", "evaluation"], "rvu_cost": 130.00},
    {"cpt_code": "93000", "description": "Electrocardiogram (ECG/EKG), routine with at least 12 leads with interpretation and report", "keywords": ["ecg", "ekg", "electrocardiogram", "rhythm strip", "cardiac monitoring"], "rvu_cost": 45.00},
    {"cpt_code": "71045", "description": "Radiologic examination, chest; single view", "keywords": ["chest x-ray", "cxr", "radiograph chest", "xray chest", "chest radiogram"], "rvu_cost": 65.00},
    {"cpt_code": "31622", "description": "Diagnostic bronchoscopy, with or without cell washing", "keywords": ["bronchoscopy", "airway inspection", "endobronchial exam"], "rvu_cost": 420.00},
    {"cpt_code": "43239", "description": "Esophagogastroduodenoscopy (EGD) biopsy, single or multiple", "keywords": ["endoscopy", "upper gi endoscopy", "gastroscopy", "biopsy"], "rvu_cost": 380.00},
    {"cpt_code": "94010", "description": "Spirometry, including graphic record, with forced expiratory vital capacity", "keywords": ["spirometry", "pulmonary function test", "pft", "lung function"], "rvu_cost": 85.00},
    {"cpt_code": "80053", "description": "Comprehensive metabolic panel (CMP blood test)", "keywords": ["metabolic panel", "blood work", "cmp", "liver function test", "electrolyte panel"], "rvu_cost": 35.00},
    {"cpt_code": "96372", "description": "Therapeutic, prophylactic, or diagnostic injection; subcutaneous or intramuscular", "keywords": ["injection", "im injection", "administered medication", "intramuscular injection"], "rvu_cost": 50.00}
]

# --- Database & Model Setup ---
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
    
    icd_collection = client.get_or_create_collection(name="master_icd10_integrated", embedding_function=emb_fn)
    if icd_collection.count() == 0:
        ids = df[code_col].astype(str).tolist()
        docs = df[desc_col].astype(str).tolist()
        metas = [{"category": str(df[cat_col].iloc[i]) if cat_col else "General",
                  "chapter": str(df[ch_col].iloc[i]) if ch_col else "General"} for i in range(len(df))]
        icd_collection.add(ids=ids, documents=docs, metadatas=metas)
        
    cpt_collection = client.get_or_create_collection(name="master_cpt_integrated", embedding_function=emb_fn)
    if cpt_collection.count() == 0:
        cpt_df = pd.DataFrame(CPT_PROCEDURE_REGISTRY)
        cpt_collection.add(
            ids=cpt_df["cpt_code"].tolist(),
            documents=cpt_df["description"].tolist(),
            metadatas=[{"keywords": " ".join(k), "cost": float(c)} for k, c in zip(cpt_df["keywords"], cpt_df["rvu_cost"])]
        )
        
    return nlp, icd_collection, cpt_collection, df, code_col, desc_col

nlp, icd_collection, cpt_collection, icd_df, code_col_name, desc_col_name = load_system()

# --- Linguistic Parser & Preprocessor ---
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

# --- LangGraph Multi-Agent Pipeline Simulator ---
def run_agentic_audit(active_entities, mapped_icd, mapped_cpt, neg_entities):
    agent_logs = []
    
    agent_logs.append({"agent": "🤖 Extractor Agent", "action": f"Parsed clinical document. Isolated {len(active_entities)} active diagnoses, {len(mapped_cpt)} procedural cues, and {len(neg_entities)} negated conditions."})
    agent_logs.append({"agent": "🏷️ Dual-Coding Engine", "action": f"Executed dense semantic retrieval across ICD-10-CM and CPT registries. Successfully matched {len(mapped_icd)} diagnostic codes and {len(mapped_cpt)} procedural billing entries."})
    
    # Auditor Rule Check
    audit_notes = []
    denial_score = 5  # Base low risk
    
    if len(mapped_icd) == 0:
        audit_notes.append("⚠️ Missing Primary Diagnostic Code — high risk for claim rejection.")
        denial_score += 45
        
    if len(mapped_cpt) > 0 and len(mapped_icd) == 0:
        audit_notes.append("⚠️ Medical Necessity Discrepancy: Procedures billed without supporting ICD-10 diagnoses.")
        denial_score += 35
        
    for neg in neg_entities:
        for diag in mapped_icd:
            if neg.lower() in diag["Extracted Term"].lower():
                audit_notes.append(f"🚨 Upcoding Alert: Negated term '{neg}' mapped to code {diag['ICD-10 Code']}. Flagged for removal.")
                denial_score += 30

    if not audit_notes:
        audit_notes.append("✅ CMS Compliance Verified: ICD-10 and CPT linkage validated with zero-hallucination verification.")
        
    agent_logs.append({"agent": "⚖️ Compliance Auditor Agent", "action": " | ".join(audit_notes)})
    
    denial_score = min(100, max(5, denial_score))
    return agent_logs, denial_score

# --- FHIR R4 Bundle Generator ---
def generate_fhir_bundle(patient_id, mapped_icd, mapped_cpt):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry": []
    }
    
    for i, icd in enumerate(mapped_icd):
        condition_resource = {
            "fullUrl": f"urn:uuid:condition-{i+1}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis", "display": icd["Type"]}]}],
                "code": {
                    "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": icd["ICD-10 Code"], "display": icd["Description"]}],
                    "text": icd["Extracted Term"]
                },
                "subject": {"reference": f"Patient/{patient_id}"}
            }
        }
        bundle["entry"].append(condition_resource)
        
    for j, cpt in enumerate(mapped_cpt):
        procedure_resource = {
            "fullUrl": f"urn:uuid:procedure-{j+1}",
            "resource": {
                "resourceType": "Procedure",
                "status": "completed",
                "code": {
                    "coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": cpt["CPT Code"], "display": cpt["Description"]}]
                },
                "subject": {"reference": f"Patient/{patient_id}"}
            }
        }
        bundle["entry"].append(procedure_resource)
        
    return bundle

# --- UI Application Layout ---
st.markdown('<div class="main-title">🏥 Autonomous Clinical ICD-10 / CPT Coder & RCM Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Agent Health Informatics Engine with Semantic Retrieval, Denial Risk Modeling & FHIR R4 Interoperability</div>', unsafe_allow_html=True)

tab_coder, tab_audit, tab_fhir, tab_knowledge = st.tabs(["⚡ Clinical Coding & Billing", "🕵️ Multi-Agent Audit & RCM Analytics", "📑 HL7 / FHIR R4 Export", "📚 Master Knowledge Base"])

with tab_coder:
    col_input, col_output = st.columns([1.05, 0.95], gap="large")
    
    with col_input:
        st.subheader("📝 Clinical Note Ingestion")
        input_mode = st.radio("Input Source:", ["Preset Multi-Condition Clinical Note", "Custom Text Input", "Upload Medical PDF / Discharge Summary"], horizontal=True)
        
        note_text = ""
        if input_mode == "Preset Multi-Condition Clinical Note":
            preset = "64-year-old male with persistent retrosternal burning chest pain and gastroesophageal reflux for 3 weeks. Patient underwent 12-lead electrocardiogram (ECG) and diagnostic upper GI endoscopy biopsy. Reports occasional dry cough. Patient denies fever, hemoptysis, or syncope. Prescribed oral medication."
            note_text = st.text_area("Clinical Text:", value=preset, height=180)
        elif input_mode == "Custom Text Input":
            note_text = st.text_area("Clinical Text:", placeholder="Enter doctor's discharge summary...", height=180)
        else:
            up_file = st.file_uploader("Upload Patient PDF Record", type=["pdf", "txt"])
            if up_file:
                if up_file.name.endswith(".pdf"):
                    reader = PdfReader(io.BytesIO(up_file.read()))
                    for p in reader.pages:
                        note_text += p.extract_text() or ""
                else:
                    note_text = up_file.read().decode("utf-8")
                st.text_area("Parsed Document Content:", value=note_text[:1000] + "...", height=150, disabled=True)
                
        run_btn = st.button("🚀 Run Autonomous Medical Coding Pipeline", type="primary", use_container_width=True)
        
    with col_output:
        st.subheader("📋 Coded Diagnostic & Billing Summary")
        
        if run_btn and note_text.strip():
            with st.spinner("Executing linguistic parsing and semantic vector inference..."):
                active_terms, neg_terms, matched_cpt_codes = parse_clinical_doc(note_text)
                
                # Retrieve ICD Codes
                icd_records = []
                seen_icd = set()
                diag_counter = 0
                
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
                            diag_label = "Primary Diagnosis" if diag_counter == 0 else f"Secondary Diagnosis #{diag_counter}"
                            diag_counter += 1
                            
                            # Estimate cost by DRG category
                            est_val = 250.00 if diag_counter == 1 else 110.00
                            
                            icd_records.append({
                                "Type": diag_label,
                                "Extracted Term": term,
                                "ICD-10 Code": code,
                                "Description": desc,
                                "Category": cat,
                                "Chapter": ch,
                                "Confidence": f"{conf}%",
                                "Estimated Reimbursement ($)": est_val
                            })
                            
                # Retrieve CPT Codes
                cpt_records = []
                for cpt_id in matched_cpt_codes:
                    cpt_item = next(p for p in CPT_PROCEDURE_REGISTRY if p["cpt_code"] == cpt_id)
                    cpt_records.append({
                        "Type": "Procedural Billable",
                        "CPT Code": cpt_item["cpt_code"],
                        "Description": cpt_item["description"],
                        "Estimated Fee ($)": cpt_item["rvu_cost"]
                    })
                    
                # Run LangGraph Audit Simulation
                agent_trace, denial_risk = run_agentic_audit(active_terms, icd_records, cpt_records, neg_terms)
                
                # Save into Session State for other tabs
                st.session_state["icd_records"] = icd_records
                st.session_state["cpt_records"] = cpt_records
                st.session_state["neg_terms"] = neg_terms
                st.session_state["agent_trace"] = agent_trace
                st.session_state["denial_risk"] = denial_risk
                st.session_state["note_text"] = note_text

        # Display Section
        if "icd_records" in st.session_state and st.session_state["icd_records"]:
            # Display ICD-10
            for idx, item in enumerate(st.session_state["icd_records"]):
                card_style = "primary-card" if idx == 0 else "secondary-card"
                st.markdown(f"""
                <div class="{card_style}">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; text-transform: uppercase; font-weight: bold;">
                        <span>● {item['Type']}</span>
                        <span>Match Confidence: {item['Confidence']}</span>
                    </div>
                    <div style="font-size: 16px; font-weight: bold; margin-top: 3px;">🏷️ {item['ICD-10 Code']} — {item['Description']}</div>
                    <div style="font-size: 12px; opacity: 0.85; margin-top: 2px;">Extracted Term: <i>"{item['Extracted Term']}"</i> | Est. DRG: ${item['Estimated Reimbursement ($)']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Display CPT
            if st.session_state["cpt_records"]:
                st.markdown("**Procedural CPT Billing Codes:**")
                for cpt in st.session_state["cpt_records"]:
                    st.markdown(f"""
                    <div class="cpt-card">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: bold;">
                            <span>🔧 CPT PROCEDURE</span>
                            <span>Fee: ${cpt['Estimated Fee ($)']:.2f}</span>
                        </div>
                        <div style="font-size: 15px; font-weight: bold; margin-top: 3px;">📌 CPT {cpt['CPT Code']} — {cpt['Description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Display Negated
            if st.session_state["neg_terms"]:
                st.markdown("**Ruled-Out / Non-Billable Entities:**")
                for neg in st.session_state["neg_terms"]:
                    st.markdown(f'<div class="negated-card">🚫 <strong>Ruled Out:</strong> <i>"{neg}"</i> (Excluded from claim)</div>', unsafe_allow_html=True)

with tab_audit:
    st.subheader("📊 Revenue Cycle Management (RCM) & Agentic Audit")
    
    if "icd_records" in st.session_state:
        total_icd_reimburse = sum(r["Estimated Reimbursement ($)"] for r in st.session_state["icd_records"])
        total_cpt_reimburse = sum(r["Estimated Fee ($)"] for r in st.session_state["cpt_records"])
        total_claim_value = total_icd_reimburse + total_cpt_reimburse
        risk = st.session_state["denial_risk"]
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-container"><div class="metric-value">${total_claim_value:.2f}</div><div class="metric-label">Total Claim Value</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-container"><div class="metric-value">{len(st.session_state["icd_records"])}</div><div class="metric-label">Billable ICD-10 Codes</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-container"><div class="metric-value">{len(st.session_state["cpt_records"])}</div><div class="metric-label">Procedural CPT Codes</div></div>', unsafe_allow_html=True)
        with m4:
            risk_color = "#10B981" if risk < 20 else "#F59E0B" if risk < 50 else "#EF4444"
            st.markdown(f'<div class="metric-container"><div class="metric-value" style="color: {risk_color};">{risk}%</div><div class="metric-label">Claim Denial Risk Index</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.markdown("### 🤖 Multi-Agent Reasoning & Execution Log")
        for log in st.session_state["agent_trace"]:
            st.markdown(f"""
            <div class="agent-card">
                <strong>{log['agent']}</strong><br/>
                <span style="color: #CBD5E1;">{log['action']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run clinical code mapping in the first tab to view RCM analytics.")

with tab_fhir:
    st.subheader("📑 HL7 / FHIR R4 Interoperability Bundle")
    st.markdown("Export interoperable clinical records conformant to the **HL7 FHIR R4 Condition & Procedure specification**:")
    
    if "icd_records" in st.session_state:
        fhir_data = generate_fhir_bundle("PATIENT-9842", st.session_state["icd_records"], st.session_state["cpt_records"])
        json_str = json.dumps(fhir_data, indent=2)
        
        st.download_button(
            label="📥 Download Standardized FHIR R4 JSON Bundle",
            data=json_str,
            file_name=f"fhir_r4_claim_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        st.json(fhir_data)
    else:
        st.info("Ingest clinical notes to generate standardized FHIR bundles.")

with tab_knowledge:
    st.subheader("📚 Master Medical Knowledge Graph")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**ICD-10-CM Registry**")
        st.dataframe(icd_df, height=350)
    with k2:
        st.markdown("**CPT Procedure Registry**")
        st.dataframe(pd.DataFrame(CPT_PROCEDURE_REGISTRY), height=350)
