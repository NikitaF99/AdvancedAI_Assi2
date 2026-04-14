import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency
import seaborn as sns
from config import *

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def generate_personas_csv(json_file_path=PARSED_PERSONAS_JSON, output_csv_path=PERSONAS_CSV):
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return None

    try:
        with open(DATA_DIR / json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Please check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return None

    personas_data = []

    for group_entry in data:
        group_num = group_entry.get("group", "N/A")
        source_model = group_entry.get("source_model", "N/A")

        for persona in group_entry.get("personas", []):
            persona_num = persona.get("persona_id", "N/A")
            name = persona.get("Name", "N/A")
            age = persona.get("Age", "N/A")
            gender = persona.get("Gender", "N/A")
            country = persona.get("Country", "N/A")

            personality_list = persona.get("Personality Traits", "N/A")
            personality = ", ".join(personality_list) if isinstance(personality_list, list) else personality_list

            technology_list = persona.get("Devices and Technologies", "N/A")
            technology = ", ".join(technology_list) if isinstance(technology_list, list) else technology_list

            experience = persona.get("Work Experience", "N/A")
            domain = persona.get("Domain of Work", "N/A")
            education = persona.get("Education", "N/A")

            personas_data.append({
                "group num": group_num,
                "source_model": source_model,
                "persona_id": persona_num,
                "name": name,
                "age": age,
                "gender": gender,
                "country": country,
                "personality": personality,
                "technology": technology,
                "experience": experience,
                "domain": domain,
                "education": education
            })

    if not personas_data:
        print("No persona data found or extracted. CSV will not be generated.")
        return None

    df = pd.DataFrame(personas_data)

    try:
        df.to_csv(output_csv_path, index=False)
        print(f"CSV file successfully generated at {output_csv_path}")
    except Exception as e:
        print(f"Error generating CSV file: {e}")
        return None

    return df


def extract_selected_persona(text, group_personas):
    if pd.isna(text):
        return None

    text_lower = str(text).lower()

    m = re.search(r"group_\d+_p[123]", text_lower)
    if m:
        return m.group(0)

    for _, row in group_personas.iterrows():
        name = str(row.get("name", "")).strip().lower()
        persona_id = str(row.get("persona_id", "")).strip()
        if name and name in text_lower:
            return persona_id

    return None


def build_final_datasets(
    personas_csv_path=PERSONAS_CSV,
    phishing_json_path=PHISHING_JSON,
    iteration_output_path=ITERATION_CSV,
    selected_only_output_path=SELECTED_ONLY_CSV,
):
    if not os.path.exists(personas_csv_path):
        raise FileNotFoundError(f"Missing file: {personas_csv_path}")

    personas_df = pd.read_csv(personas_csv_path)
    personas_df.columns = [c.strip().lower() for c in personas_df.columns]
    personas_df = personas_df.rename(columns={"group num": "group_num"})

    if "name" not in personas_df.columns:
        personas_df["name"] = ""

    personas_df["group_num"] = personas_df["group_num"].astype(int)
    personas_df["source_model"] = personas_df["source_model"].astype(str).str.strip()
    personas_df["persona_id"] = personas_df["persona_id"].astype(str).str.strip()

    if not os.path.exists(phishing_json_path):
        raise FileNotFoundError(f"Missing file: {phishing_json_path}")

    with open(DATA_DIR / phishing_json_path, "r", encoding="utf-8") as f:
        phishing_data = json.load(f)

    decision_rows = []

    for entry in phishing_data:
        evaluator_model = str(entry.get("model", "")).strip()
        group_num = int(entry.get("group_id"))
        iteration = int(entry.get("iteration"))
        reason = str(entry.get("analysis_output", "")).strip()

        # Match personas by group only
        group_personas = personas_df[
            personas_df["group_num"] == group_num
        ]

        selected_persona_id = extract_selected_persona(reason, group_personas)

        decision_rows.append({
            "evaluator_model": evaluator_model,
            "group_num": group_num,
            "iteration": iteration,
            "selected_persona_id": selected_persona_id,
            "reason": reason
        })

    decisions_df = pd.DataFrame(decision_rows)

    expanded_rows = []

    for _, decision in decisions_df.iterrows():
        evaluator_model = decision["evaluator_model"]
        group_num = decision["group_num"]
        iteration = decision["iteration"]
        selected_persona_id = decision["selected_persona_id"]
        reason = decision["reason"]

        group_personas = personas_df[
            personas_df["group_num"] == group_num
        ]

        for _, persona in group_personas.iterrows():
            row = persona.to_dict()
            row["evaluator_model"] = evaluator_model
            row["iteration"] = iteration
            row["selected_persona_id"] = selected_persona_id
            row["phishing_susceptible"] = (
                "Yes" if str(persona["persona_id"]) == str(selected_persona_id) else "No"
            )
            row["reason"] = reason if row["phishing_susceptible"] == "Yes" else "N/A"
            expanded_rows.append(row)

    iteration_df = pd.DataFrame(expanded_rows)
    iteration_df.to_csv(iteration_output_path, index=False)
    iteration_df[iteration_df["phishing_susceptible"] == "Yes"].to_csv(selected_only_output_path, index=False)

    print(f"Saved: {iteration_output_path}")
    print(f"Saved: {selected_only_output_path}")

    return iteration_df


def load_iteration_dataframe(csv_path=ITERATION_CSV):
    df = pd.read_csv(csv_path)
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["selected"] = df["phishing_susceptible"].map({"Yes": 1, "No": 0})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def map_domain(domain):
    d = str(domain).lower()

    if any(k in d for k in [
        "software", "data", "ai", "machine learning", "tech",
        "it", "robotics", "virtual reality", "developer"
    ]):
        return "Technology"
    elif any(k in d for k in [
        "business", "marketing", "finance", "manager",
        "entrepreneur"
    ]):
        return "Business"
    elif any(k in d for k in [
        "education", "research", "scientist"
    ]):
        return "Education"
    elif any(k in d for k in [
        "social", "community", "pharmacist"
    ]):
        return "Social"
    elif any(k in d for k in [
        "design", "art", "curator"
    ]):
        return "Creative"
    else:
        return "Other"


def map_region(country):
    c = str(country).strip().lower()

    if c in ["india", "japan", "south korea", "china", "singapore"]:
        return "Asia"
    elif c in ["uae", "united arab emirates", "saudi arabia", "lebanon"]:
        return "Middle East"
    elif c in ["sweden", "germany", "uk", "england"]:
        return "Europe"
    elif c in ["usa", "united states", "canada", "mexico"]:
        return "North America"
    elif c in ["brazil"]:
        return "South America"
    elif c in ["kenya", "ghana", "nigeria", "morocco"]:
        return "Africa"
    else:
        return "Other"


def age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    else:
        return "35+"


def run_gender_bias_analysis(csv_path=ITERATION_CSV):
    df = load_iteration_dataframe(csv_path)

    gender_bias = df.groupby("gender")["selected"].mean().sort_values(ascending=False)
    print("Gender bias:")
    print(gender_bias)
    print()

    subset = df[df["gender"].isin(["female", "male"])]
    table = pd.crosstab(subset["gender"], subset["selected"])
    print(table)

    if table.shape == (2, 2):
        odds_ratio, p = fisher_exact(table)
        print("p-value:", p)
        print("odds_ratio:", odds_ratio)
    else:
        print("Fisher test skipped: table is not 2x2.")

    return gender_bias


def run_domain_bias_analysis(csv_path=ITERATION_CSV):
    df = load_iteration_dataframe(csv_path)
    df["domain_group"] = df["domain"].apply(map_domain)

    grouped_domain_bias = (
        df.groupby("domain_group")["selected"]
        .mean()
        .sort_values(ascending=False)
    )

    print("Domain bias:")
    print(grouped_domain_bias)
    print()

    table = pd.crosstab(df["domain_group"], df["selected"])
    print(table)
    print()

    chi2, p, dof, expected = chi2_contingency(table)
    print("Chi-square:", chi2)
    print("p-value:", p)

    return grouped_domain_bias


def run_region_bias_analysis(csv_path=ITERATION_CSV):
    df = load_iteration_dataframe(csv_path)
    df["region"] = df["country"].apply(map_region)

    region_bias = (
        df.groupby("region")["selected"]
        .mean()
        .sort_values(ascending=False)
    )

    print("Region bias:")
    print(region_bias)
    print()

    table = pd.crosstab(df["region"], df["selected"])
    print(table)
    print()

    chi2, p, dof, expected = chi2_contingency(table)
    print("Chi-square:", chi2)
    print("p-value:", p)

    return region_bias


def run_age_bias_analysis(csv_path=ITERATION_CSV):
    df = load_iteration_dataframe(csv_path)
    df["age_group"] = df["age"].apply(age_group)

    table = pd.crosstab(df["age_group"], df["selected"])
    print(table)
    print()

    chi2, p, dof, expected = chi2_contingency(table)
    print("Chi-square:", chi2)
    print("p-value:", p)
    print()

    age_group_stats = df.groupby("age_group").agg(
        total_count=("selected", "count"),
        selected_count=("selected", "sum"),
        selection_rate=("selected", "mean")
    )

    print(age_group_stats)
    return age_group_stats


def plot_gender_domain_heatmap(csv_path=ITERATION_CSV):
    df = load_iteration_dataframe(csv_path)
    df["domain_group"] = df["domain"].apply(map_domain)

    gender_domain_bias = df.groupby(["gender", "domain_group"])["selected"].mean().unstack()

    plt.figure(figsize=(8, 5))
    sns.heatmap(gender_domain_bias, annot=True, cmap="coolwarm")
    plt.title("Gender vs Domain Selection Rate")
    plt.tight_layout()
    plt.show()

    return gender_domain_bias


def main():
    generate_personas_csv()
    build_final_datasets()
    run_gender_bias_analysis()
    run_domain_bias_analysis()
    run_region_bias_analysis()
    run_age_bias_analysis()
    plot_gender_domain_heatmap()


if __name__ == "__main__":
    main()