import psutil
import time

def get_running_processes():
    processes = []
    for process in psutil.process_iter(['pid', 'name', 'status']):
        processes.append({
            'pid': process.info['pid'],
            'name': process.info['name'],
            'status': process.info['status']
        })
    return processes

def monitor_process_progress(process_name):
    # این تابع یک شبیه‌سازی از پیشرفت پردازش انجام می‌دهد.
    # برای واقعی‌تر کردن، این بخش را با کد واقعی جایگزین کنید.
    progress = 0
    while progress < 100:
        time.sleep(1)
        progress += 10
        print(f"پیشرفت پردازش '{process_name}': {progress}%")

if __name__ == "__main__":
    # اسم پردازه‌ای که می‌خواهید نظارت کنید (بر اساس اسم فایل اجرایی)
    target_process_name = "python.exe"

    # نمایش اطلاعات پردازه‌های در حال اجرا
    running_processes = get_running_processes()
    print("پردازه‌های در حال اجرا:")
    for process in running_processes:
        print(f"PID: {process['pid']}, Name: {process['name']}, Status: {process['status']}")

    # جستجو برای پردازه با نام موردنظر
    target_process = next((process for process in running_processes if process['name'] == target_process_name), None)

    if target_process:
        # شروع نظارت بر پیشرفت پردازه موردنظر
        monitor_process_progress(target_process_name)
    else:
        print(f"پردازه با نام '{target_process_name}' یافت نشد.")