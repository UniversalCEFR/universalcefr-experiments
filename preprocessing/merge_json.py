import json

def merge_json_files(json_files, output_file):
    """
    Merges multiple JSON files into one JSON file.

    Parameters:
        json_files (list): List of JSON file paths to merge.
        output_file (str): Path to the output JSON file.
    """
    merged_data = []

    for file_name in json_files:
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Ensure the data is a list before merging
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: File {file_name} does not contain a list. Skipping.")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing file {file_name}: {e}")

    # Write the merged data to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, indent=4, ensure_ascii=False)

# Example usage
json_files = ["A2.json","B1.json"]  # Replace with your JSON file names
output_file = "apa-lha.json"  # Replace with your desired output file name
merge_json_files(json_files, output_file)
