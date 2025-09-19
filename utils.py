# utils.py
import os
import pandas as pd

STUDENT_FILE = "students.csv"

def ensure_companies_loaded(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please place companies.csv in project root.")
    return pd.read_csv(path)

def save_student_profile(record, filepath=STUDENT_FILE):
    """
    Save or update a student profile in the CSV file.

    Args:
        record (dict): Student profile as a dictionary with column names as keys.
        filepath (str): Path to the CSV file.

    Returns:
        dict: The saved/updated student record.
    """
    # Load existing dataframe or create new
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=record.keys())

    # Ensure all record keys exist in dataframe columns
    for key in record.keys():
        if key not in df.columns:
            df[key] = ""

    email = record.get("email", None)
    if not email:
        raise ValueError("Record must contain an 'email' field to identify student.")

    # If student already exists → update
    if "email" in df.columns and email in df["email"].values:
        for k, v in record.items():
            df.loc[df["email"] == email, k] = v
    else:
        # Append new record
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    # Save back to CSV
    df.to_csv(filepath, index=False)

    return record