import json
import pandas as pd

#  Define the constant system message for all students
SYSTEM_PROMPT = """The student demonstrates a strong understanding of basic arithmetic operations and pattern recognition.
However, they struggle with multi-step problems and geometry-related questions.
They are likely to succeed on straightforward procedural problems but may falter when required to apply concepts creatively or abstractly."""

#  Load your dataset
input_csv = "processed_prompting_data.csv"  # Update the path if needed
df = pd.read_csv(input_csv)

# Group questions by student
student_to_questions = df.groupby("UserId")["QuestionId"].apply(list).to_dict()

#  Build persona-style prompts
persona_prompts = []
for student_id, question_ids in student_to_questions.items():
    if len(question_ids) < 3:
        continue  # Skip if fewer than 3 questions

    selected_questions = question_ids[:3]  # Take the first 3 questions
    user_prompt = "Based on this persona, predict whether the student will answer the following questions correctly:\n"
    for qid in selected_questions:
        user_prompt += f"- Question {qid}:\n"
    user_prompt += "\nRespond with 'Likely Correct' or 'Likely Incorrect' for each."

    persona_prompts.append({
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "student_id": student_id,
        "question_ids": selected_questions
    })

#  Save prompts to JSON and CSV
output_json = "persona_prompts.json"
output_csv = "persona_prompts.csv"

with open(output_json, "w") as f:
    json.dump(persona_prompts, f, indent=2)
print(f" Saved prompts to {output_json}")

pd.DataFrame(persona_prompts).to_csv(output_csv, index=False)
print(f" Also saved prompts to {output_csv}")
