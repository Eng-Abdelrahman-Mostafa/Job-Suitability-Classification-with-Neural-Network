import pandas as pd
import numpy as np

data = pd.read_csv("job_dataset_without_interview.csv")
# Define weights for interview performance
weights = {
    "course_grades": 0.25,
    "projects_completed": 0.25,
    "experience_years": 0.34,
    "extracurriculars": 0.1,
    "is_suitable": 0.06
}

# Normalize columns
data['normalized_is_suitable'] = data['is_suitable']
data['interview_score'] = (
    data['course_grades'] / 100 * weights['course_grades'] +
    data['projects_completed'] / data['projects_completed'].max() * weights['projects_completed'] +
    data['experience_years'] / data['experience_years'].max() * weights['experience_years'] +
    data['extracurriculars'] / data['extracurriculars'].max() * weights['extracurriculars'] +
    data['normalized_is_suitable'] * weights['is_suitable']
)

# Scale the score to 0–5 range
data['interview_performance'] = (data['interview_score'] / data['interview_score'].max()) * 5
data['interview_performance'] += np.random.uniform(-0.2, 0.2, size=len(data))
data['interview_performance'] = data['interview_performance'].clip(0, 5)

# Drop intermediate columns
data.drop(columns=['normalized_is_suitable', 'interview_score'], inplace=True)

# Save the updated dataset
output_file = "Updated_Data_with_Interview_Performance.csv"
data.to_csv(output_file, index=False)

print(f"File saved as {output_file}")