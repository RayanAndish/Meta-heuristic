import os
import pandas as pd

# مسیر دایرکتوری حاوی فایل‌های CSV
directory_path = r'D:\test'

# گردش در دایرکتوری و خواندن تمام فایل‌های CSV
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    if filename.endswith(".csv") and os.path.isfile(file_path):
        # خواندن داده‌ها از فایل CSV
        df = pd.read_csv(file_path)

        # حذف ردیف‌های خالی
        df_cleaned = df.dropna(how='all')

        # ذخیره فایل CSV پاک‌شده (جایگزین فایل اصلی)
        df_cleaned.to_csv(file_path, index=False)

print("تمامی فایل‌های CSV در دایرکتوری پاک‌سازی شدند.")
