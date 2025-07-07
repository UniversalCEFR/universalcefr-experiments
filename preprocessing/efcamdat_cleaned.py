import pandas as pd
import json

# Paths to the input Excel file and output JSON file
input_xlsx_path = 'database.xlsx'
output_json_path = 'efcamdat-cleaned.json'

# Read the Excel file
df = pd.read_excel(input_xlsx_path)

# Select only the "cefr" and "text_corrected" columns
filtered_df = df[['cefr', 'text_corrected']]

# Transform the DataFrame into the desired JSON format
transformed_data = []

for _, row in filtered_df.iterrows():
    cefr_level = row['cefr'] if pd.notna(row['cefr']) else 'Unknown'  # Default to 'Unknown' if missing
    text_content = row['text_corrected'] if pd.notna(row['text_corrected']) else ''  # Default to empty string if missing

    transformed_entry = {
        "title": "na",  # Assuming 'na' as per your example
        "lang": "en",  # Assuming English language
        "source_name": "efcamdat-cleaned",
        "format": "paragraph-level",  # Assuming paragraph-level format
        "category": "learner",
        "cefr_level": str(cefr_level.strip()),
        "license": "Cambridge",
        "text": str(text_content.strip())
    }

    transformed_data.append(transformed_entry)

# Write the transformed data to the output JSON file
with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(transformed_data, jsonfile, indent=4, ensure_ascii=False)

print(f'Transformed data has been written to {output_json_path}')