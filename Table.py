import os
import pandas as pd

def read_csv_files(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df.insert(0, 'File_Name', filename)
            dataframes.append(df)
    return dataframes

def create_excel_table(dataframes, output_file):
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    merged_df.to_excel(output_file, index=False)

if __name__ == "__main__":
    input_folder = "D:/table"  # تغییر مسیر به پوشه فایل‌های CSV خود
    output_excel_file = "D:/Table/result.xlsx"  # تغییر مسیر به فایل اکسل خروجی

    csv_dataframes = read_csv_files(input_folder)
    create_excel_table(csv_dataframes, output_excel_file)

    print(f"The Excel file has been created: {output_excel_file}")
