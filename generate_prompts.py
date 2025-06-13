import pandas as pd
import os

# --- Step 0: Load and merge data ---
df1 = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\train_data\train_task_1_2.csv")
df2 = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\train_data\train_task_3_4.csv")

train_df = pd.concat([df1, df2]).reset_index(drop=True)
print(f"Total records in combined training data: {len(train_df)}")

# --- Step 1: Pick a student with enough answers ---
student_counts = train_df['UserId'].value_counts()
students_with_enough_data = student_counts[student_counts >= 10]

if students_with_enough_data.empty:
    raise ValueError("No student with 10+ answers found.")

selected_user = students_with_enough_data.index[0]
print(f"Selected student ID: {selected_user}")

# --- Step 2: Split data into persona/test sets ---
student_data = train_df[train_df['UserId'] == selected_user].reset_index(drop=True)

persona_df = student_data[:-7]  # use all but last 7 for persona
test_df = student_data[-7:]     # last 7 for testing

# --- Step 3: Build persona prompt ---
def create_persona_prompt(df):
    prompt = "The student answered the following questions:\n\n"
    for _, row in df.iterrows():
        q_id = row['QuestionId']
        correctness = "Correct" if row['IsCorrect'] == 1 else "Incorrect"
        prompt += f"- Question {q_id}: {correctness}\n"
    prompt += "\nBased on this, describe the student's learning persona, strengths, and weaknesses."
    return prompt

persona_prompt = create_persona_prompt(persona_df)

# --- Step 4: Placeholder persona (replace this in playground later) ---
placeholder_persona = """
The student demonstrates a strong understanding of basic arithmetic operations and pattern recognition.
However, they struggle with multi-step problems and geometry-related questions.
They are likely to succeed on straightforward procedural problems but may falter when required to apply concepts creatively or abstractly.
"""

def create_prediction_prompt(persona_text, df):
    prompt = f"Here is a student persona:\n{persona_text}\n\n"
    prompt += "Based on this persona, predict whether the student will answer the following questions correctly:\n"
    for _, row in df.iterrows():
        prompt += f"- Question {row['QuestionId']}:\n"
    prompt += "\nRespond with 'Likely Correct' or 'Likely Incorrect' for each."
    return prompt

prediction_prompt = create_prediction_prompt(placeholder_persona, test_df)

# --- Step 5: Save prompts ---
output_dir = "student_prompts_v2"
os.makedirs(output_dir, exist_ok=True)

persona_file_path = os.path.join(output_dir, f"persona_prompt_user_{selected_user}.txt")
prediction_file_path = os.path.join(output_dir, f"prediction_prompt_user_{selected_user}.txt")

with open(persona_file_path, "w") as f:
    f.write(persona_prompt)

with open(prediction_file_path, "w") as f:
    f.write(prediction_prompt)

print(f"✅ Prompts saved for user {selected_user} in '{output_dir}'.")

# --- Step 6: Save actual answers ---
answers_file_path = os.path.join(output_dir, f"actual_answers_user_{selected_user}.txt")
with open(answers_file_path, "w") as f:
    f.write(f"Actual answers for student {selected_user}:\n\n")
    for _, row in test_df.iterrows():
        qid = row["QuestionId"]
        correct = "Correct" if row["IsCorrect"] == 1 else "Incorrect"
        f.write(f"Question {qid}: {correct}\n")

print(f"✅ Actual answers saved to '{answers_file_path}'")
