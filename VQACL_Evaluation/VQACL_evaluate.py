import os
import re
import pandas as pd
import numpy as np

# Update this path to target your raw file
FILE_PATH = "InternVL2-26B_VQACL_results.xlsx"
OUTPUT_PATH = "InternVL2-26B_VQACL_Clean_Scores.xlsx"
SUMMARY_PATH = "InternVL2-26B_VQACL_acc.csv"
def strict_clean(value):
    """
    Converts to string, drops capitalization, strips whitespaces,
    and cleanly slices away brackets/quotes if they wrap the text.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    # Strip away python list formatting artifacts like ['abc'] or ["abc"]
    text = re.sub(r"[\[\]\'\"]", "", text)
    return text.strip()

def print_clean_results(df, score_col='eval_score_fixed'):
    """
    Prints a beautiful, structured accuracy report to the terminal.
    """
    overall_acc = df[score_col].mean() * 100
    
    # Header
    print("\n" + "="*45)
    print("            EVALUATION REPORT            ")
    print("="*45)
    print(f" {'OVERALL ACCURACY'.ljust(25)} | {overall_acc:.2f}%")
    print("-" * 45)
    
    # Category Breakdown
    if 'category' in df.columns:
        # Group, calculate mean, and sort by name
        cat_scores = df.groupby('category')[score_col].mean() * 100
        for cat, score in cat_scores.items():
            print(f" {str(cat).ljust(25)} | {score:.2f}%")
    else:
        print(" [Warning: No 'category' column found in data]")
        
    print("="*45 + "\n")

def run_comparison():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print("Loading file...")
    df = pd.read_excel(FILE_PATH)

    # Determine ground truth column name
    gt_col = 'answers' if 'answers' in df.columns else 'answer'
    if gt_col not in df.columns or 'prediction' not in df.columns:
        print(f"Missing required columns! Found: {list(df.columns)}")
        return

    print("Processing accurate matching values (1 or 0)...")
    
    accurate_matches = []
    
    for _, row in df.iterrows():
        # Clean both the ground truth and the model's raw string output
        clean_gt = strict_clean(row[gt_col])
        clean_pred = strict_clean(row['prediction'])
        
        # Binary assignment: 1 for exact match, 0 for mismatch
        if clean_pred == clean_gt and clean_gt != "":
            accurate_matches.append(1)
        else:
            accurate_matches.append(0)

    # Inject the clean values back into your sheet columns
    df['eval_match_fixed'] = accurate_matches
    df['eval_score_fixed'] = accurate_matches

    # Save out the raw results back to the folder
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Successfully evaluated! File saved containing fixed metrics: {OUTPUT_PATH}")
    
    # Quick percentage stdout check
    total_rows = len(df)
    total_correct = sum(accurate_matches)
    print(f"Processed: {total_rows} items.")
    print(f"Accurate Match Rate: {(total_correct / total_rows) * 100:.2f}%")

    # Assuming 'df' is your final DataFrame with the fixed 'eval_score_fixed' column...

    # 1. Initialize a summary dictionary with all categories defaulted to 0.0
    summary_metrics = {
        "Overall": np.mean(df['eval_score_fixed']) * 100,
        "action": 0.0, "causal": 0.0, "color": 0.0, "commonsense": 0.0,
        "count": 0.0, "judge": 0.0, "location": 0.0, "recognition": 0.0,
        "subcategory": 0.0, "type": 0.0
    }

    # 2. Compute accurate category percentages where they exist in the data
    if 'category' in df.columns:
        cat_averages = df.groupby('category')['eval_score_fixed'].mean() * 100
        for cat in cat_averages.index:
            if cat in summary_metrics:
                summary_metrics[cat] = round(cat_averages[cat], 2)

    # 3. Form a single-row DataFrame and dump directly to CSV
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    # Print the clean evaluation results
    print_clean_results(df, score_col='eval_score_fixed')

run_comparison()