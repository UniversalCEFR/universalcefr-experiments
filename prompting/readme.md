### Prompting LLMs for CEFR Level Classification using UniversalCEFR

The code for prompting is fairly simple and can accept command-type args via the terminal.

For example, the code below uses Gemma7b (English) using language-specific reading comprehension prompts:

'python script.py --prompt_template prompt_files/lang_read/prompt_with_specs.txt --dataset_id UniversalCEFR/universalcefr_test --lang_prompts --model_id google/gemma-7b-it'

One the other hand, if you want to Gemma7b (English) using English-only reading comprehension prompts:

'python script.py --prompt_template prompt_files/lang_read/prompt_with_specs_en.txt --dataset_id UniversalCEFR/universalcefr_test --model_id google/gemma-7b-it'

Just change 'lang_read' to 'lang_write' if you want to use written production prompts.
