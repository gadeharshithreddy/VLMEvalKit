import pandas as pd
import ast

# 1. Load your file
file_path = r"outputs/InternVL2-26B/T20260623-150728/InternVL2-26B_VQAv2DatasetSubset.xlsx"
df = pd.read_excel(file_path)

# 2. Define the standard VQA evaluation metric rule
def calculate_vqa_score(prediction, answers_list_str):
    try:
        # Parse the string-encoded list of dicts safely
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

# 3. Dynamic Column Fallbacks (Prevents KeyError)
pred_col = 'prediction' if 'prediction' in df.columns else 'Prediction'
answers_list_col = 'answers' if 'answers' in df.columns else 'answer'

# 4. Calculate score for every row
df['vqa_score'] = df.apply(lambda row: calculate_vqa_score(row[pred_col], row[answers_list_col]), axis=1)

# 5. Generate and display the summary reports
print("============ FINAL SUMMARY REPORT ============\n")
print(f"Overall VQA Accuracy: {df['vqa_score'].mean() * 100:.2f}%\n")

if 'answer_type' in df.columns:
    print("------------ Score by Answer Type ------------")
    answer_types = df.groupby('answer_type')['vqa_score'].mean() * 100
    for name, score in answer_types.items():
        print(f"{name:<20} : {score:.2f}%")

if 'question_type' in df.columns:
    print("\n------------ Score by Question Type ------------")
    question_types = df.groupby('question_type')['vqa_score'].mean() * 100
    for name, score in question_types.items():
        print(f"{name:<20} : {score:.2f}%")