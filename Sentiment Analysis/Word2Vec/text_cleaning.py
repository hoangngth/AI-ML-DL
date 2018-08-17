import re
input_file = open("raw.txt", "r", encoding="utf8")
output_file = open("clean.txt", "w", encoding="utf8")

def clean_text(text):
    text = re.sub(r'[^\w]', ' ', text) # Remove all symbols
    text = re.sub(r'^\s+' , '' , text) # Remove leading whitespace
    text = re.sub(r' +',' ', text) # Remove all duplicate white spaces
    return text

print("Cleaning text...")
for line in input_file:
    cleaned_text = clean_text(line)
    if not cleaned_text.strip(): continue # Skip the line if it is empty
    output_file.write(cleaned_text)
    output_file.write("\n")
    
input_file.close()
output_file.close()
print("Clean text completed!")