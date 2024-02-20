import csv

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        for row in reader:
            # Process each row if necessary, for example, handling triple quotes
            new_row = [field.replace('"""', '"') for field in row]  # Simplistic handling
            writer.writerow(new_row)

input_path = 'test_llama7.csv'  # Adjust to your file path
output_path = 'cleaned_dataset7.csv'
clean_csv(input_path, output_path)