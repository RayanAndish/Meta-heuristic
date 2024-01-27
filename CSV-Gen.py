import pandas as pd
import random

# تعیین تعداد ردیف‌ها
num_rows = 2000

# لیست‌های مربوط به داده‌های تصادفی
names = ['علی', 'محمد', 'فاطمه', 'زهرا', 'امیر', 'سارا', 'رضا', 'نازنین', 'پارسا', 'نیلوفر', 'علی‌رضا', 'سمانه']
last_names = ['احمدی', 'رضایی', 'کریمی', 'محمدی', 'فرهنگ', 'نجفی', 'صداقت', 'میرزایی', 'علیزاده', 'شجاعی']
schools = ['دبیرستان علامه حلی', 'دبیرستان فرزانگان', 'دبیرستان امام حسین', 'دبیرستان ابن‌سینا', 'دبیرستان شهید بهشتی']
grades = list(range(8, 16))
gpas = [round(random.uniform(15.50, 20.00), 2) for _ in range(num_rows)]
subject_grades = [round(random.uniform(15.00, 20.00), 2) for _ in range(num_rows)]

# ایجاد دیتافریم
df = pd.DataFrame({
    'نام و نام خانوادگی': [f'{random.choice(names)} {random.choice(last_names)}' for _ in range(num_rows)],
    'نام مدرسه': [random.choice(schools) for _ in range(num_rows)],
    'سن': [random.randint(8, 15) for _ in range(num_rows)],
    'جنسیت': [random.choice(['پسر', 'دختر']) for _ in range(num_rows)],
    'معدل کل سال اول ابتدایی': gpas,
    'معدل کل سال دوم ابتدایی': gpas,
    'معدل کل سال سوم ابتدایی': gpas,
    'معدل کل سال چهارم ابتدایی': gpas,
    'معدل کل سال پنجم ابتدایی': gpas,
    'معدل کل سال ششم ابتدایی': gpas,
    'معدل کل سال هفتم': gpas,
    'معدل کل سال هشتم': gpas,
    'معدل کل سال نهم': gpas,
    'مقطع فعلی': [random.choice(['ابتدایی', 'متوسطه']) for _ in range(num_rows)],
    'نمره درس ریاضی': subject_grades,
    'نمره درس علوم': subject_grades,
    'نمره درس حرفه': subject_grades,
    'نمره درس 1': subject_grades,
    'نمره درس 2': subject_grades,
    'نمره درس 3': subject_grades,
    'نمره درس 4': subject_grades,
    'نمره درس 5': subject_grades,
    'نمره درس 6': subject_grades,
})

# ذخیره دیتافریم در یک فایل CSV
df.to_csv('student_data.csv', index=False, encoding='utf-8-sig')

print("فایل CSV با موفقیت ایجاد شد.")