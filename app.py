# app.py
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
from recommender import Recommender
from resume_parser import parse_resume_with_llm
from utils import save_student_profile, STUDENT_FILE, ensure_companies_loaded

# Setup
st.set_page_config(layout="wide", page_title="PM Internship Matcher (Streamlit)")
st.title("PM Internship Matcher — Student Portal")

DATA_DIR = Path(".")
COMPANIES_CSV = DATA_DIR / "companies.csv"
STUDENTS_CSV = DATA_DIR / STUDENT_FILE

# Ensure companies file exists (raises helpful error)
companies_df = ensure_companies_loaded(COMPANIES_CSV)

# instantiate recommender (loads model & precomputes embeddings)
recommender = Recommender(companies_df)

# Sidebar: navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Register (Manual)", "Upload Resume", "Company Preview", "About"])

# Common inputs
def build_student_record_from_inputs(name, email, education, skills, sector_interests, location_pref, experience, is_rural, source):
    return {
        "name": name.strip(),
        "email": email.strip(),
        "education": education.strip(),
        "skills": skills.strip(),
        "sector_interests": sector_interests.strip(),
        "location_pref": location_pref.strip(),
        "experience": experience.strip(),
        "is_rural": bool(is_rural),
        "source": source,
        "timestamp": datetime.utcnow().isoformat()
    }

# Page: Manual registration
if page == "Register (Manual)":
    st.header("Manual Registration")
    st.markdown("Enter your details and get top internship recommendations. Your profile will be saved (students.csv).")

    with st.form("manual_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full name")
            email = st.text_input("Email")
            education = st.text_input("Highest education (e.g., B.Tech, 2nd year)")
            sector_interests = st.text_input("Sector interests (comma separated)")
            is_rural = st.checkbox("I belong to a rural/tribal background")
        with col2:
            skills = st.text_area("Skills (comma separated)", help="e.g., Python, SQL, Excel")
            location_pref = st.text_input("Preferred location (city) or 'remote'")
            experience = st.text_input("Experience (e.g., 'Fresher' or '2 years')")
            submit = st.form_submit_button("Save & Recommend")

        if submit:
            # build record
            record = build_student_record_from_inputs(
                name, email, education, skills, sector_interests, location_pref, experience, is_rural, "manual"
            )

            # save
            saved = save_student_profile(record)
            st.success(f"Profile saved for {saved['name']} ({saved['email']})")

            # run recommender
            candidate_text = " ".join([saved["education"], saved["skills"], saved["sector_interests"], saved["experience"]])
            candidate_skills = saved["skills"]
            results = recommender.recommend(candidate_text, candidate_skills, candidate_location_pref=saved["location_pref"], is_rural=saved["is_rural"], top_k=5)

            st.subheader("Top recommendations")
            for r in results:
                st.markdown(f"**{r['PostedRole']} @ {r['CompanyName']}** — {r['Location']} — Stipend: {r.get('Stipend','N/A')}")
                st.write(f"Required skills: {r.get('SkillsRequired','')}")
                st.metric("Match score", f"{round(r['score']*100,1)}%")
                st.markdown("---")

# Page: Resume upload
elif page == "Upload Resume":
    st.header("Upload Resume")
    st.markdown("Upload a PDF resume — we will extract profile using LLM (Groq) and show recommendations. Your profile will be saved to students.csv.")

    uploaded_file = st.file_uploader("Upload resume (PDF, TXT)", type=["pdf", "txt"])
    is_rural = st.checkbox("I belong to a rural/tribal background (optional)")
    if uploaded_file:
        with st.spinner("Extracting text & parsing resume (this may take a few seconds)..."):
            raw_bytes = uploaded_file.read()
            try:
                parsed_profile = parse_resume_with_llm(raw_bytes)
            except Exception as e:
                st.error("LLM parsing failed: " + str(e))
                parsed_profile = {
                    "name": "", "email": "", "education": "",
                    "skills": "", "sector_interests": "", "location_pref": "", "experience": ""
                }

        # allow user edit/confirm
        st.subheader("Extracted profile (please confirm / edit)")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", parsed_profile.get("name",""))
            email = st.text_input("Email", parsed_profile.get("email",""))
            education = st.text_input("Education", parsed_profile.get("education",""))
            sector_interests = st.text_input("Sector interests", parsed_profile.get("sector_interests",""))
        with col2:
            skills = st.text_area("Skills (comma separated)", parsed_profile.get("skills",""))
            location_pref = st.text_input("Location preference", parsed_profile.get("location_pref",""))
            experience = st.text_input("Experience", parsed_profile.get("experience",""))
            
        if st.button("Save & Recommend"):
            record = build_student_record_from_inputs(
                name, email, education, skills, sector_interests, location_pref, experience, is_rural, "resume"
            )
            saved = save_student_profile(record)
            st.success(f"Profile saved for {saved['name']} ({saved['email']})")

            # recommend
            candidate_text = " ".join([saved["education"], saved["skills"], saved["sector_interests"], saved["experience"]])
            results = recommender.recommend(candidate_text, saved["skills"], candidate_location_pref=saved["location_pref"], is_rural=saved["is_rural"], top_k=5)
            st.subheader("Top recommendations")
            for r in results:
                st.markdown(f"**{r['PostedRole']} @ {r['CompanyName']}** — {r['Location']} — Stipend: {r.get('Stipend','N/A')}")
                st.write(f"Required skills: {r.get('SkillsRequired','')}")
                st.metric("Match score", f"{round(r['score']*100,1)}%")
                st.markdown("---")

# Page: Company Preview (admin)
elif page == "Company Preview":
    st.header("Company Preview — Registered Students")
    st.markdown("This preview shows `students.csv` so companies can see registered students and their skills.")
    if STU := Path(STUDENTS_CSV).exists():
        df = pd.read_csv(STUDENTS_CSV)
        st.write(f"Total registered: {len(df)}")
        st.dataframe(df)
        st.markdown("You can copy this CSV or download it:")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download students.csv", csv, "students.csv", "text/csv")
    else:
        st.info("No students registered yet.")

# About
else:
    st.header("About")
    st.markdown(
        """
        Overview

    The Internship Matcher is a prototype platform that leverages AI + ML to connect students with suitable internship opportunities.
    It automatically parses student resumes or accepts manually entered profiles, then recommends internships from a company database using semantic similarity + rule-based features.

    This project demonstrates how AI can bridge the gap between students and opportunities, especially for those in rural or underrepresented areas.

    Features

    📄 Resume Parsing — Uses Groq LLM (Llama 3.1) to extract structured student profiles from resumes.

    📝 Manual Profile Entry — Students can enter details directly if they don’t have a resume ready.

    🤖 Fallback Mode — If LLM fails, a rule-based extractor ensures parsing still works.

    🧠 Smart Recommendations — Internship matching powered by SentenceTransformers embeddings + Jaccard skill similarity.

    ⚖️ Weighted Scoring — Matches consider:

    Semantic similarity (roles, skills, industry)

        Jaccard skill overlap

        Location preference

        Government/NGO bonus (for rural students)

    💾 Data Handling —

        Students are saved in students.csv.

        Companies are read from companies.csv.

    🚀 Streamlit Frontend — Interactive student portal demo with two input modes:
            """
    )
