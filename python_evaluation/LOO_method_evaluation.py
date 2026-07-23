import pandas as pd
import ast
import os

INPUT_FILE_NAME = "./Gemma4-26B-A4B-it_VQACP5000_results.xlsx"  # Change this to your actual file name
OUTPUT_FILE_NAME = "./Gemma4-26B-A4B-it_VQACP5000_results_updated.xlsx"  # Output file name
OUTPUT_CSV_NAME = "./Gemma4-26B-A4B-it_VQACP5000_results.csv"  # Output CSV file name
def normalize_text(text):
    """Converts text to lowercase and strips whitespace/punctuation."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip().strip(".")

def calculate_vqa_score(prediction, ground_truths_str):
    """Calculates the official VQA metric and generates a binary match list."""
    try:
        # Safely parse the string representation of the list
        truths_list = ast.literal_eval(ground_truths_str)
        if not isinstance(truths_list, list):
            return 0.0, []
    except (ValueError, SyntaxError):
        return 0.0, []

    # Normalize prediction and all ground truths
    norm_pred = normalize_text(prediction)
    norm_truths = [normalize_text(ans) for ans in truths_list]

    # Create a list of 1s (match) and 0s (mismatch) corresponding to each ground truth
    match_list = [1 if norm_pred == truth else 0 for truth in norm_truths]
    
    # Count total exact matches
    matches = sum(match_list)
    
    # Apply official VQA formula: min(1.0, matches / 3.0)
    score = min(1.0, matches / 3.0)
    return score, match_list

def main(input_excel_path):
    print(f"Loading {input_excel_path}...")
    
    # Load the Excel file
    try:
        df = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # --- THE FIX: Convert the column to float so it accepts decimals ---
    if 'eval_score' in df.columns:
        df['eval_score'] = pd.to_numeric(df['eval_score'], errors='coerce').astype(float)
    # -------------------------------------------------------------------

    # Process each row
    for index, row in df.iterrows():
        prediction = row['prediction']
        ground_truths = row['answers']
        original_score = row['eval_score']
        
        # Calculate new VQA score and get the list of 1s and 0s
        new_score, match_list = calculate_vqa_score(prediction, ground_truths)
        
        # Write the list of matches to eval_match
        df.at[index, 'eval_match'] = str(match_list)
        
        # Replace the eval_score if the new VQA score is higher
        if pd.isna(original_score):
             df.at[index, 'eval_score'] = new_score
        elif new_score > original_score:
             df.at[index, 'eval_score'] = new_score

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