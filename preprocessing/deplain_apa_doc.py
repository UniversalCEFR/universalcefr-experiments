import pandas as pd
import json

# Paths to the input Excel file and output JSON file
input_xlsx_path = 'all.csv'
output_json_path = 'deplain-apa-complex.json'

# Read the Excel file
df = pd.read_csv(input_xlsx_path)

# Select only the "cefr" and "text_corrected" columns
filtered_df = df[['original', 'complex_level', 'license', 'complex_title']]

# Transform the DataFrame into the desired JSON format
transformed_data = []

for _, row in filtered_df.iterrows():
    cefr_level = row['complex_level'] if pd.notna(row['complex_level']) else 'Unknown'  # Default to 'Unknown' if missing
    text_content = row['original'] if pd.notna(row['original']) else ''  # Default to empty string if missing
    license = row['license'] if pd.notna(row['license']) else ''
    title = row['complex_title'] if pd.notna(row['complex_title']) else ''

    transformed_entry = {
        "title": str(title.strip()),  # Assuming 'na' as per your example
        "lang": "de",  # Assuming English language
        "source_name": "deplain-apa-doc",
        "format": "document-level",  # Assuming paragraph-level format
        "category": "reference",
        "cefr_level": str(cefr_level.strip().upper()),
        "license": str(license.strip()),
        "text": str(text_content.strip())
    }

    transformed_data.append(transformed_entry)

# Write the transformed data to the output JSON file
with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(transformed_data, jsonfile, indent=4, ensure_ascii=False)

print(f'Transformed data has been written to {output_json_path}')