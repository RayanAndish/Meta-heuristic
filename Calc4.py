import os
import pandas as pd
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor

# مسیر دایرکتوری حاوی فایل‌های CSV
directory_path = 'D:/Results-v/'

# لیست تمام فایل‌های CSV در دایرکتوری
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# استفاده از تنها یک فایل
single_csv_file = csv_files[0]

# تابع پردازش یک فایل CSV
def process_csv(file_name):
    print(f"شروع پردازش فایل {file_name}")
    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path, converters={'Best_Solutions': literal_eval})
    result_df = pd.DataFrame()

    for column in df.columns:
        if column != 'Iteration':
            if df[column].apply(type).eq(list).any():
                column_values = [val for sublist in df[column] for val in sublist]
                result_df[column + '_Mean'] = [sum(column_values) / len(column_values)]
                result_df[column + '_Minimum'] = [min(column_values)]
                result_df[column + '_Maximum'] = [max(column_values)]
                result_df[column + '_Variance'] = [sum((x - (sum(column_values) / len(column_values))) ** 2 for x in column_values) / len(column_values)]
                result_df[column + '_Standard_Deviation'] = [(sum((x - (sum(column_values) / len(column_values))) ** 2 for x in column_values) / len(column_values)) ** 0.5]

    result_df['Best_Fitness_Mean'] = df['Best_Fitness'].mean()
    result_df['Best_Fitness_Minimum'] = df['Best_Fitness'].min()
    result_df['Best_Fitness_Maximum'] = df['Best_Fitness'].max()
    result_df['Best_Fitness_Variance'] = df['Best_Fitness'].var()
    result_df['Best_Fitness_Standard_Deviation'] = df['Best_Fitness'].std()

    result_df['Execution_Time_Mean'] = df['Execution_Time'].mean()
    result_df['Execution_Time_Minimum'] = df['Execution_Time'].min()
    result_df['Execution_Time_Maximum'] = df['Execution_Time'].max()
    result_df['Execution_Time_Variance'] = df['Execution_Time'].var()
    result_df['Execution_Time_Standard_Deviation'] = df['Execution_Time'].std()

    # ذخیره نتایج به یک فایل csv
    output_file_name = file_name.replace('.csv', '-v.csv')
    output_file_path = os.path.join(directory_path, output_file_name)
    result_df.to_csv(output_file_path, index=False)

    print(f"پردازش فایل {output_file_name} به اتمام رسید.")

# استفاده از شرط محافظت
if __name__ == '__main__':
    # ایجاد یک ProcessPoolExecutor برای اجرای موازی با حداکثر تعداد هسته‌های فیزیکی
    with ProcessPoolExecutor() as executor:
        # استفاده از تابع process_csv بر روی یک فایل به صورت موازی
        executor.submit(process_csv, single_csv_file)

    print("پردازش به اتمام رسید.")