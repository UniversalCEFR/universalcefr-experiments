import os
import json

# Path to the folder containing the .txt files
txt_folder_path = 'A2-OR'

# Function to extract the text content from the .txt file
def extract_text_from_file(txt_file):
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # Return the entire content including the first line
        content = ''.join(lines).strip()
        return content, lines  # Return content and lines for length check
    except UnicodeDecodeError:
        # If UTF-8 fails, try a different encoding (e.g., latin-1)
        with open(txt_file, 'r', encoding='latin-1') as file:
            lines = file.readlines()
        # Return the entire content including the first line
        content = ''.join(lines).strip()
        return content, lines

# Create a list to hold all the JSON objects
json_objects = []

# Loop through all .txt files in the folder
for txt_file in os.listdir(txt_folder_path):
    if txt_file.endswith('.simpde'): #and 'A2' in txt_file:
        # Get the full path of the txt file
        txt_file_path = os.path.join(txt_folder_path, txt_file)

        # Extract the text content and lines from the file
        text_content, lines = extract_text_from_file(txt_file_path)

        # Create the JSON object
        json_obj = {
            "title": str(txt_file),
            "lang": "de",  #
            "source_name": "apa-lha",
            "format": "document-level",
            "category": "reference",
            "cefr_level": "A2",  # Assuming A2 for all files, can be adjusted if needed
            "license": "Unknown",
            "text": str(text_content)
        }

        # Add the JSON object to the list
        json_objects.append(json_obj)

# Output the JSON objects to a file
output_json_path = 'A2.json'
with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(json_objects, jsonfile, indent=4, ensure_ascii=False)

print(f'JSON data has been written to {output_json_path}')
