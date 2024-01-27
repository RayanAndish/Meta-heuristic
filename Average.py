import csv
import sys

# Set the field size limit manually
csv.field_size_limit(2**31 - 1)

def calculate_average(numbers_str):
    numbers = [float(num) for num in numbers_str.strip('[]').split(',')]
    return sum(numbers) / len(numbers)

def transform_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        for row in rows:
            for i, cell in enumerate(row):
                if cell.startswith('[') and cell.endswith(']'):
                    row[i] = str(calculate_average(cell))

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Specify the input CSV file path here
input_file = 'AVOA-Ackely-D10-25000.csv'

# Specify the output CSV file path here
output_file = 'AVOA-Ackely-D10-25000-AV.csv'

transform_csv(input_file, output_file)