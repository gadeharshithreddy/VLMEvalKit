import pandas as pd
import os

INPUT_FILE_NAME = "Gemma4-26B-A4B-it_COCOGQA_results.xlsx"  # Change this to your actual file name
OUTPUT_FILE_NAME = "Gemma4-26B-A4B-it_COCOGQA_results_updated.xlsx"
OUTPUT_CSV_NAME = "Gemma4-26B-A4B-it_COCOGQA_results.csv"

def normalize_text(text):
    """Converts text to lowercase and strips whitespace/punctuation."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip().strip(".")

def main(input_excel_path):
    print(f"Loading {input_excel_path}...")
    
    # Load the Excel file
    try:
        df = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Coerce eval_score to float to avoid TypeErrors with int64 columns
    if 'eval_score' in df.columns:
        df['eval_score'] = pd.to_numeric(df['eval_score'], errors='coerce').astype(float)

    # Process each row
    for index, row in df.iterrows():
        ground_truth = row.get('answer', "") 
        prediction = row.get('prediction', row.get('predictior', "")) 
        original_score = row.get('eval_score', pd.NA)

        norm_truth = normalize_text(ground_truth)
        norm_pred = normalize_text(prediction)
        
        # Simple Exact Match Logic
        is_match = (norm_truth == norm_pred and norm_truth != "")
        
        if is_match:
            match_str = "[1.0]"
        else:
            match_str = "[0.0]"
            
        # Write the match string back to the dataframe
        df.at[index, 'eval_match'] = match_str
        
        # --- THE CORRECTED LOGIC ---
        # If the script finds a perfect match, but the original score was 0 (or missing)
        # we upgrade the score to 1.0. Otherwise, we leave the original score untouched.
        if is_match and (pd.isna(original_score) or original_score < 1.0):
             df.at[index, 'eval_score'] = 1.0
        # -------------------------

    # 1. Save the updated Excel file
    output_excel_name = OUTPUT_FILE_NAME
    df.to_excel(output_excel_name, index=False)
    print(f"Saved updated itemized results to: {output_excel_name}")

    # 2. Calculate overall accuracy and save to CSV
    total_questions = len(df)
    sum_of_scores = df['eval_score'].sum()
    overall_accuracy = (sum_of_scores / total_questions) * 100 if total_questions > 0 else 0

    csv_data = {
        "Total_Questions": [total_questions],
        "Sum_of_Scores": [sum_of_scores],
        "Overall_Accuracy_Percentage": [round(overall_accuracy, 2)]
    }
    
    summary_df = pd.DataFrame(csv_data)
    output_csv_name = OUTPUT_CSV_NAME
    summary_df.to_csv(output_csv_name, index=False)
    print(f"Saved overall accuracy summary to: {output_csv_name}")
    print(f"Final Accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    # Ensure you have 'openpyxl' installed (pip install pandas openpyxl)
    input_file = INPUT_FILE_NAME # Change this to your actual file name
    
    if os.path.exists(input_file):
        main(input_file)
    else:
        print(f"Please update the 'input_file' variable. Could not find: {input_file}")