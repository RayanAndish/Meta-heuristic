from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# خواندن داده‌های آموزشی
data = pd.read_csv('D:/Algurithms/student_data2.csv')
X_train = data[['معدل کل سال پنجم ابتدایی', 'معدل کل سال ششم ابتدایی', 'نمره درس ریاضی', 'نمره درس علوم']]
y_train = data[['معدل کل سال نهم']]

# آموزش مدل یادگیری ماشین
model = LinearRegression()
model.fit(X_train, y_train)

# ذخیره مدل
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        student_name = request.form['student_name']
        gpa_5th = float(request.form['gpa_5th'])
        gpa_6th = float(request.form['gpa_6th'])
        math_score = float(request.form['math_score'])
        science_score = float(request.form['science_score'])

        # ایجاد دیتافریم جهت پیش‌بینی
        student_data = pd.DataFrame({'معدل کل سال پنجم ابتدایی': [gpa_5th],
                                     'معدل کل سال ششم ابتدایی': [gpa_6th],
                                     'نمره درس ریاضی': [math_score],
                                     'نمره درس علوم': [science_score]})

        # بارگذاری مدل
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # پیش‌بینی
        result = model.predict(student_data)
        result_text = f"نتیجه پیش‌بینی برای {student_name}: {result[0]}"

        return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)