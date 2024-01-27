import os
import multiprocessing

def run_algorithm(process_id):
    # اجرای فایل algorithm.py در هر پردازه
    os.system("python D:/Algurithms/BWOA-Ackley.py")

if __name__ == "__main__":
    num_cores = 24

    # اجرای توابع مورد نظر به صورت موازی
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(run_algorithm, range(num_cores))

    print("All processes completed.")