import linecache
import subprocess
import sys

import tokenizers


def get_tokenizer() -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer.from_pretrained("bert-base-chinese")


class Lines:
    def __init__(self, filename, *, skip=0, group=1, preserve_newline=False):
        self.filename = filename
        with open(filename):
            pass
        if sys.platform == "win32":
            with open(filename) as f:
                linecount = sum(1 for _ in f)
        else:
            output = subprocess.check_output(("wc -l " + filename).split())
            linecount = int(output.split()[0])
        self.length = (linecount - skip) // group
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        d = self.skip + 1
        if isinstance(item, int):
            if item < len(self):
                if self.group == 1:
                    line = linecache.getline(self.filename, item + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [linecache.getline(self.filename, d + item * self.group + k) for k in range(self.group)]
                    if not self.preserve_newline:
                        line = [li.strip("\r\n") for li in line]
                return line

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = _clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = _clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                if self.group == 1:
                    line = linecache.getline(self.filename, i + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [linecache.getline(self.filename, d + i * self.group + k) for k in range(self.group)]
                    if not self.preserve_newline:
                        line = [li.strip("\r\n") for li in line]
                ls.append(line)

            return ls

        raise IndexError


def _clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v
