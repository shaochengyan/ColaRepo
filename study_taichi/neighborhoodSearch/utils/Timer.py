import time

class Timer:
    def __init__(self, name=None) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"{self.name}: 代码执行时间：{elapsed_time}秒")

if __name__=="__main__":
    with Timer():
        print("!")
    