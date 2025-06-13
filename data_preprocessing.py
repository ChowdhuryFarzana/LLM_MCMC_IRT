import pandas as pd

# Load data
train_df = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\train_data\train_task_1_2.csv")
student_df = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\metadata\student_metadata_task_1_2.csv")
question_df = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\metadata\question_metadata_task_1_2.csv")
answer_df = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\metadata\answer_metadata_task_1_2.csv")
subject_df = pd.read_csv(r"C:\Users\chowd23f\Documents\LLM_research\data\data\metadata\subject_metadata.csv")


# Merge to include SubjectId
train_merged = train_df.merge(question_df[['QuestionId', 'SubjectId']], on='QuestionId', how='left')

# Merge with student metadata
train_merged = train_merged.merge(student_df, on='UserId', how='left')

# Calculate student accuracy
accuracy_df = train_merged.groupby('UserId')['IsCorrect'].mean().reset_index()
accuracy_df.columns = ['UserId', 'PercentageCorrect']
train_merged = train_merged.merge(accuracy_df, on='UserId', how='left')

# Fix dtype mismatch before merging with subject metadata
train_merged['SubjectId'] = train_merged['SubjectId'].astype(str)
subject_df['SubjectId'] = subject_df['SubjectId'].astype(str)

# Merge subject name
train_merged = train_merged.merge(subject_df[['SubjectId', 'Name']], on='SubjectId', how='left')

# Select 10 random students for test cases
selected_students = train_merged['UserId'].drop_duplicates().sample(10, random_state=42)
filtered_df = train_merged[train_merged['UserId'].isin(selected_students)]

# Keep useful columns for prompting
final_df = filtered_df[[
    'UserId', 'PercentageCorrect', 'SubjectId', 'Name',
    'QuestionId', 'IsCorrect', 'Gender'
]]

# Save to CSV
final_df.to_csv("processed_prompting_data.csv", index=False)
print("Processed data saved as 'processed_prompting_data.csv'")