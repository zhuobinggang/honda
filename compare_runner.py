from taku_subword_expand import run as run1
from taku_title import run as run2

def old():
    # 重复实验1和2
    run1(seed = 10, indexs = range(0, 3))
    run1(seed = 10, indexs = range(0, 3))
    run2(seed = 10, indexs = range(3, 10)) # NOTE: 失误，因为种子一样，所以3~6的结果是重复的
    run2(seed = 10, indexs = range(3, 10)) # NOTE: 失误，因为种子一样，所以3~6的结果是重复的

# 补充重复实验3
def now():
    run1(seed = 11, indexs = range(3, 6))
    run2(seed = 11, indexs = range(3, 6))
