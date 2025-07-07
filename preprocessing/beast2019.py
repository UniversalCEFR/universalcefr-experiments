import json

# Path to the input JSON file
input_json_path = 'B.dev.json'
# Path to the output JSON file
output_json_path = 'B.dev.new.json'

# List to hold the transformed data
transformed_data = []

# Read the input JSON file and handle multiple JSON objects
with open(input_json_path, 'r', encoding='utf-8') as infile:
    # Read the entire file content
    content = infile.read().strip()

    # Split the content by newlines to get individual JSON objects
    entries = content.split('\n')

    # Ensure each entry is valid JSON and add commas between them
    valid_entries = []
    for entry in entries:
        entry = entry.strip()  # Remove any extra spaces
        if entry:  # Skip empty lines
            try:
                # Parse each individual JSON object
                valid_entries.append(json.loads(entry))
            except json.JSONDecodeError as e:
                print(f"Error decoding entry: {e}")
                continue

# Loop through the loaded data and transform it
for entry in valid_entries:
    # Extract the necessary fields
    cefr_level = entry.get('cefr', 'Unknown')  # Default to 'Unknown' if no CEFR level is provided
    text_content = entry.get('text', '')

    # Create the new JSON structure
    transformed_entry = {
        "title": "na",  # Assuming 'na' as per your example
        "lang": "en",  # Assuming English language
        "source_name": "bea2019st-write-improve",
        "format": "sentence-level",  # Assuming sentence-level format
        "category": "learner",
        "cefr_level": cefr_level[:2],
        "license": "CC BY-SA-NC 4.0",
        "text": text_content
    }

    # Append the transformed entry to the list
    transformed_data.append(transformed_entry)

# Write the transformed data to the output JSON file
with open(output_json_path, 'w', encoding='utf-8') as outfile:
    json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)

print(f'Transformed data has been written to {output_json_path}')