import psutil
import os
import cv2
import gc
import time


# img = cv2.imread("/home/yhao/Pictures/uitest/debug/mmexport1720429785599.png")
process = psutil.Process(os.getpid())
print(f"Memory before: {process.memory_info().rss / 1e6} MB")
# img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
# start = 0
# while start < 10:
#     gc.collect()
#     time.sleep(1)
#     start += 1
print(f"Memory after: {process.memory_info().rss / 1e6} MB")