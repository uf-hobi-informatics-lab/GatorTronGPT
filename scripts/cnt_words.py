import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor


def load_lines(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
    return lines


def _cnt(lines):
    tmp = 0
    for line in lines:
        text = json.loads(line)['text']
        tmp += len(text.split())
    return tmp


def get_cnt(fn):
    temp_cnt = 0
    lines = load_lines(sys.argv[1] + "/" + fn)

    step = len(lines) // 32 + 1

    with ProcessPoolExecutor(max_workers=32) as pool:
        for each in pool.map(_cnt, [lines[i, i+step] for i in range(0, len(lines), step)]):
            temp_cnt += each
        
    return temp_cnt


def main():
    files = [f for f in os.listdir(sys.argv[1]) if f.endswith(".json")]
    cnt = 0
    for each in map(get_cnt, files):
        cnt += each

    print(cnt)


if __name__ == '__main__':
    main()
