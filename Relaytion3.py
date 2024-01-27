import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(input_csv_path):
    # خواندن فایل CSV و تبدیل آن به یک DataFrame
    df = pd.read_csv(input_csv_path)

    # تبدیل ستون 'date' به تاریخ
    df['date'] = pd.to_datetime(df['date'])

    # نمودار توزیع هر ستون
    df.plot(kind='box', subplots=True, layout=(2, 3), sharex=False, sharey=False)
    plt.suptitle('Distribution of Columns')
    plt.show()

    # نمودار همبستگی
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Matrix')
    fig.colorbar(cax)
    plt.show()

# مثال استفاده:
input_file_path = 'D:/Ehram/input.csv'  # مسیر فایل ورودی
visualize_data(input_file_path)