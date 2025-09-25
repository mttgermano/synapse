import os
import re
import time
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from google import genai

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 4
INITIAL_BACKOFF = 1.0
INPUT_CSV = "evaluate/rag_responses.csv"
OUTPUT_CSV = "evaluate/rag_metrics.csv"

def parse_float_from_text(text):
    if not text: return None
    text_fixed = text.replace(",", ".")
    matches = re.findall(r"0(?:\.\d+)?|1(?:\.0+)?", text_fixed)
    if not matches:
        m2 = re.findall(r"\.\d+", text_fixed)
        if m2:
            try:
                val = float(m2[0])
                if 0.0 <= val <= 1.0:
                    return val
            except:
                return None
        return None
    try:
        val = float(matches[0])
        if 0.0 <= val <= 1.0:
            return val
    except:
        return None
    return None

def ask_gemini_for_score(client, model, prompt, max_retries=MAX_RETRIES):
    backoff = INITIAL_BACKOFF
    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            raw = ""
            if hasattr(resp, "text") and resp.text is not None:
                raw = resp.text
            else:
                raw = str(resp)
            score = parse_float_from_text(raw)
            return score, raw, None
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff)
            backoff *= 2
    return None, None, f"Failed after {max_retries} attempts. Last error: {last_err}"

def build_prompts(row):
    q = row.get("question")
    ctx = row.get("contexts")
    ans = row.get("answer")
    prompt_relevancy = (
        "You are an evaluator. Given a user query, a set of context documents, and a generated answer, "
        "provide a single numeric score between 0.0 and 1.0 that represents ANSWER RELEVANCY: "
        "how well the answer actually addresses the user's QUERY (0 = not relevant at all, 1 = fully relevant). "
        "Return only the numeric score (optionally with brief justification on next line)."
        f"\n\nQuery: {q}\n\nContext: {ctx}\n\nAnswer: {ans}\n\nScore (0-1):"
    )
    prompt_faithfulness = (
        "You are an evaluator. Given a set of context documents and a generated answer, "
        "provide a single numeric score between 0.0 and 1.0 that represents FAITHFULNESS: "
        "how much of the answer is supported by the provided contexts (0 = contradicted or hallucinated, 1 = fully supported). "
        "Return only the numeric score (optionally with brief justification on next line)."
        f"\n\nContext: {ctx}\n\nAnswer: {ans}\n\nScore (0-1):"
    )
    return prompt_relevancy, prompt_faithfulness

def main():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("API_KEY"))
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    out_cols = ["answer_relevancy", "faithfulness"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""
    for idx in tqdm(df.index, desc="Avaliação RAG"):
        row = df.loc[idx].to_dict()
        if row.get("answer_relevancy") not in ("", None) and row.get("faithfulness") not in ("", None):
            continue

        prompt_rel, prompt_fa = build_prompts(row)

        rel_score, rel_raw, rel_err = ask_gemini_for_score(client, MODEL_NAME, prompt_rel)
        time.sleep(0.2)
        fa_score, fa_raw, fa_err = ask_gemini_for_score(client, MODEL_NAME, prompt_fa)

        df.at[idx, "answer_relevancy"] = "" if rel_score is None else f"{rel_score:.4f}"
        df.at[idx, "faithfulness"] = "" if fa_score is None else f"{fa_score:.4f}"

        df.to_csv(OUTPUT_CSV, index=False)

    print("Avaliação finalizada. Resultado em:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
