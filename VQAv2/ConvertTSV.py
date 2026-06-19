import json
from pathlib import Path

import pandas as pd
from vlmeval.smp.vlm import encode_image_file_to_base64

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

QUESTIONS_FILE = "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FILE = "v2_mscoco_val2014_annotations.json"
IMAGE_DIR = Path("val2014")

OUTPUT_TSV = "vqav2_val.tsv"

# ------------------------------------------------------------------
# Load source files
# ------------------------------------------------------------------

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)["questions"]

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]

# ------------------------------------------------------------------
# Build question_id -> annotation mapping
# ------------------------------------------------------------------

ann_map = {
    ann["question_id"]: ann
    for ann in annotations
}

# ------------------------------------------------------------------
# Convert to TSV rows
# ------------------------------------------------------------------

rows = []

for q in questions:
    qid = q["question_id"]
    image_id = q["image_id"]

    ann = ann_map[qid]

    image_filename = f"COCO_val2014_{image_id:012d}.jpg"
    image_path = IMAGE_DIR / image_filename

    row = {
        # Common benchmark fields
        "index": qid,
        "question_id": qid,
        "image_id": image_id,
        "question": q["question"],

        # Annotation fields
        "multiple_choice_answer": ann["multiple_choice_answer"],
        "question_type": ann["question_type"],
        "answer_type": ann["answer_type"],

        # Preserve complete answer list
        "answers": json.dumps(
            ann["answers"],
            ensure_ascii=False
        ),

        # Preserve raw image in VLMEvalKit format
        "image": encode_image_file_to_base64(str(image_path)),
    }

    rows.append(row)

# ------------------------------------------------------------------
# Save TSV
# ------------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_TSV,
    sep="\t",
    index=False
)

print(f"Saved {len(df):,} rows to {OUTPUT_TSV}")