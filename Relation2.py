import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocess_and_save(input_csv_path, processed_csv_path, output_csv_path):
    # خواندن فایل CSV و تبدیل آن به یک DataFrame
    df = pd.read_csv(input_csv_path)

    # تبدیل ستون 'date' به تاریخ
    df['date'] = pd.to_datetime(df['date'])

    # ایجاد یک ستون جدید برای نمایش تفاوت روزها
    df['day_diff'] = (df['date'] - df['date'].shift(1)).dt.days.fillna(0)

    # ایجاد ستون‌های ویژگی (X) و متغیر پاسخ (y)
    X = df[['open', 'high', 'low', 'close', 'day_diff']]
    y = df['close'].shift(-1)

    # جدا کردن داده‌ها به داده‌های آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ایجاد مدل رگرسیون خطی
    model = LinearRegression()
    model.fit(X_train, y_train)

    # پیش‌بینی اعداد روز بعد
    df['close_next_day_pred'] = model.predict(X)

    # حذف ستون‌های اضافی
    df = df.drop(['day_diff'], axis=1)

    # ذخیره کردن داده‌های پردازش شده در فایل میانی
    df.to_csv(processed_csv_path, index=False)

    # محاسبه میانگین مربعات خطا
    mse = mean_squared_error(y[~np.isnan(y)], df['close_next_day_pred'][:-1][~np.isnan(y)])
    print(f'Mean Squared Error: {mse}')

    # ذخیره کردن نتایج در یک فایل CSV جدید
    df.to_csv(output_csv_path, index=False)

# مثال استفاده:
input_file_path = 'D:/Ehram/input.csv'  # مسیر فایل ورودی
processed_file_path = 'D:/Ehram/processed_output.csv'  # مسیر فایل خروجی پردازش شده
output_file_path = 'D:/Ehram/final_output.csv'  # مسیر فایل خروجی

preprocess_and_save(input_file_path, processed_file_path, output_file_path)
