import pandas as pd
import ast
import os

# 1. Path to your prediction file
file_path = r"outputs\InternVL2-26B\T20260623-150728\InternVL2-26B_VQAv2DatasetSubset.xlsx"
df = pd.read_excel(file_path)

# 2. Define standard VQA evaluation metric rule
def calculate_vqa_score(prediction, answers_list_str):
    try:
        answers = ast.literal_eval(answers_list_str)
    except:
        return 0.0
    
    pred = str(prediction).strip().lower()
    
    match_count = 0
    for ans_dict in answers:
        if str(ans_dict.get('answer', '')).strip().lower() == pred:
            match_count += 1
            
    return min(match_count / 3, 1.0)

# Determine structural columns dynamically
pred_col = 'prediction' if 'prediction' in df.columns else 'Prediction'
answers_list_col = 'answers' if 'answers' in df.columns else 'answer'

# 3. Calculate scores
df['vqa_score'] = df.apply(lambda row: calculate_vqa_score(row[pred_col], row[answers_list_col]), axis=1)

# 4. Compute metrics dictionary matching VLMEvalKit's naming format
metrics_dict = {}

# Overall Score
metrics_dict['Overall'] = round(df['vqa_score'].mean() * 100, 2)

# Answer Type Scores
if 'answer_type' in df.columns:
    answer_types = df.groupby('answer_type')['vqa_score'].mean() * 100
    for name, score in answer_types.items():
        metrics_dict[name] = round(score, 2)

# Question Type Scores
if 'question_type' in df.columns:
    question_types = df.groupby('question_type')['vqa_score'].mean() * 100
    for name, score in question_types.items():
        metrics_dict[name] = round(score, 2)

# 5. Convert to horizontal single-row DataFrame (VLMEvalKit structure)
output_df = pd.DataFrame([metrics_dict])

# 6. Automatically resolve destination path in the same folder
# Changes extension from .xlsx to _acc.csv
base_path, _ = os.path.splitext(file_path)
output_csv_path = base_path + "_acc.csv"

# Save the file
output_df.to_csv(output_csv_path, index=False)

print("============ VLMEvalKit Format Output ============")
print(f"Summary file successfully saved to:\n{output_csv_path}\n")
print(output_df.to_string(index=False))