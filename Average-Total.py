import csv

# Set the field size limit manually
csv.field_size_limit(2**31 - 1)

def calculate_average(numbers_str):
    numbers = [float(num) for num in numbers_str.strip('[]').split(',')]
    return sum(numbers) / len(numbers)

def transform_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Calculate column averages
        column_sums = [0.0] * len(rows[0])
        for row in rows:
            for i, cell in enumerate(row):
                if cell.startswith('[') and cell.endswith(']'):
                    numbers = [float(num) for num in cell.strip('[]').split(',')]
                    column_sums[i] += sum(numbers)

        column_averages = [sums / len(rows) for sums in column_sums]

        # Replace cells with row averages
        for row in rows:
            for i, cell in enumerate(row):
                if cell.startswith('[') and cell.endswith(']'):
                    row[i] = str(calculate_average(cell))

        # Replace the last row with column averages
        rows[-1] = [str(avg) for avg in column_averages]

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Specify the input CSV file path here
input_file = 'AVOA-Ackely-D10-100.csv'

# Specify the output CSV file path here
output_file = 'AVOA-Ackely-D10-100-AV.csv'

transform_csv(input_file, output_file)