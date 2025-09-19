# recommender.py
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = os.getenv("SENT_MODEL", "all-MiniLM-L6-v2")

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_skills(text):
    text = normalize_text(text)
    parts = re.split(r"[,;\n]| and ", text)
    skills = [p.strip() for p in parts if p.strip()]
    return list(dict.fromkeys(skills))

def jaccard_score(list1, list2):
    s1 = set([normalize_text(x) for x in list1 if x])
    s2 = set([normalize_text(x) for x in list2 if x])
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

def _normalize_embeddings(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms

class Recommender:
    def __init__(self, companies_df):
        self.companies = companies_df.copy()
        for c in self.companies.columns:
            self.companies[c] = self.companies[c].fillna("")
        self.companies['combined_text'] = (
            self.companies['SkillsRequired'].astype(str) + " " +
            self.companies['PostedRole'].astype(str) + " " +
            self.companies['Industry'].astype(str)
        )
        # auto-detect IsGovernment from CompanyName or Sector
        gov_keywords = ["ngo","gov","government","public sector","psu","co-op","cooperative","council","trust","rural development","community development"]
        def detect_gov(row):
            t = f"{row.get('CompanyName','')} {row.get('Sector','')}".lower()
            return any(k in t for k in gov_keywords)
        self.companies['IsGovernment'] = self.companies.apply(detect_gov, axis=1)

        # load model and compute embeddings
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        texts = self.companies['combined_text'].apply(normalize_text).tolist()
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = _normalize_embeddings(emb)

    def recommend(self, candidate_text, candidate_skills, candidate_location_pref=None, is_rural=False, top_k=5,
                  weights={'embed':0.7, 'jaccard':0.2, 'location':0.05, 'gov':0.05}):
        cand_emb = self.model.encode([normalize_text(candidate_text)], convert_to_numpy=True, show_progress_bar=False)
        cand_emb = _normalize_embeddings(cand_emb)[0]
        sim = np.dot(self.embeddings, cand_emb)  # cosine since normalized

        jaccard = self.companies['SkillsRequired'].apply(lambda r: jaccard_score(parse_skills(candidate_skills), parse_skills(r))).values

        if candidate_location_pref:
            loc = normalize_text(candidate_location_pref)
            location_bonus = self.companies['Location'].apply(lambda r: 1.0 if loc and loc in normalize_text(r) else 0.0).values
        else:
            location_bonus = np.zeros(len(self.companies))

        if is_rural:
            gov_bonus = self.companies['IsGovernment'].apply(lambda x: 1.0 if x else 0.0).values
        else:
            gov_bonus = np.zeros(len(self.companies))

        final_score = (weights['embed'] * sim) + (weights['jaccard'] * jaccard) + (weights['location'] * location_bonus) + (weights['gov'] * gov_bonus)

        res = self.companies.copy()
        res['score'] = final_score
        res = res.sort_values('score', ascending=False).head(top_k)
        # convert to list of dicts
        return res.to_dict(orient="records")
