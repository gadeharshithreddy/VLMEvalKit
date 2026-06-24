import pandas as pd
import ast
import os

# 1. Load your file
file_path = r"outputs\InternVL2-26B\T20260623-150728\InternVL2-26B_VQAv2DatasetSubset.xlsx"
df = pd.read_excel(file_path)

# 2. Define the standard VQA evaluation metric rule
def calculate_vqa_score(prediction, answers_list_str):
    try:
        answers = ast.literal_eval(answers_list_str)
    except:
        return 0.0
    
    pred = str(prediction).strip().lower()
    
    # Count how many human reviewers provided this exact answer
    match_count = 0
    for ans_dict in answers:
        if str(ans_dict.get('answer', '')).strip().lower() == pred:
            match_count += 1
            
    # VQA Metric formula: min(match_count / 3, 1.0)
    return min(match_count / 3, 1.0)

# Dynamic Column Fallbacks (Prevents KeyError)
pred_col = 'prediction' if 'prediction' in df.columns else 'Prediction'
answers_list_col = 'answers' if 'answers' in df.columns else 'answer'

# 3. Calculate score for every row
df['vqa_score'] = df.apply(lambda row: calculate_vqa_score(row[pred_col], row[answers_list_col]), axis=1)

# 4. Generate metrics for internal tracking
metrics_dict = {'Overall': round(df['vqa_score'].mean() * 100, 2)}

# Terminal printing buffers
answer_print_lines = []
question_print_lines = []

if 'answer_type' in df.columns:
    answer_stats = df.groupby('answer_type')['vqa_score'].agg(['mean', 'size'])
    for name, row in answer_stats.iterrows():
        score_pct = row['mean'] * 100
        count = int(row['size'])
        metrics_dict[name] = round(score_pct, 2)
        answer_print_lines.append(f"| {name:<20} | {count:<8} | {score_pct:>7.2f}% |")

if 'question_type' in df.columns:
    question_stats = df.groupby('question_type')['vqa_score'].agg(['mean', 'size'])
    for name, row in question_stats.iterrows():
        score_pct = row['mean'] * 100
        count = int(row['size'])
        metrics_dict[name] = round(score_pct, 2)
        question_print_lines.append(f"| {name:<25} | {count:<8} | {score_pct:>7.2f}% |")

# 5. Automatically save the file in the identical VLMEvalKit summary structure
# output_df = pd.DataFrame([metrics_dict])
# base_path, _ = os.path.splitext(file_path)
# output_csv_path = base_path + "_acc.csv"
# output_df.to_csv(output_csv_path, index=False)


# 6. Beautiful Terminal Output
print("\n" + "="*53)
print(f"│ SUMMARY REPORT: {os.path.basename(file_path)} │")
print("="*53)
print(f" Overall Accuracy : {metrics_dict['Overall']:.2f}%")
print(f" Total Questions  : {len(df)}")
print("-"*53)

if answer_print_lines:
    print("\n[ Performance by Answer Type ]")
    print("+" + "-"*22 + "+" + "-"*10 + "+" + "-"*11 + "+")
    print(f"| {'Category':<20} | {'Count':<8} | {'Accuracy':<9} |")
    print("+" + "-"*22 + "+" + "-"*10 + "+" + "-"*11 + "+")
    for line in answer_print_lines:
        print(line)
    print("+" + "-"*22 + "+" + "-"*10 + "+" + "-"*11 + "+")

if question_print_lines:
    print("\n[ Performance by Question Type ]")
    print("+" + "-"*27 + "+" + "-"*10 + "+" + "-"*11 + "+")
    print(f"| {'Prefix Structure':<25} | {'Count':<8} | {'Accuracy':<9} |")
    print("+" + "-"*27 + "+" + "-"*10 + "+" + "-"*11 + "+")
    for line in question_print_lines:
        print(line)
    print("+" + "-"*27 + "+" + "-"*10 + "+" + "-"*11 + "+")

print("\n" + "="*53)
# print(f"💾 File successfully saved right beside targets at:\n   {os.path.basename(output_csv_path)}")
# print("="*53 + "\n")