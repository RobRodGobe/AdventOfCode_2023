import sys
import re
from functools import reduce
from collections import defaultdict
from typing import List, Tuple
import math

def day_1():
    # Create a variable to read files
    filename = "Files/Day_1.txt"
    total = 0

    # Open file and split into "rows" (lines)
    with open(filename) as f:
        lines = f.readlines()

        # Loop through each line
        for line in lines:
            digits = re.sub(r'\D', '', line)  # Remove non-digit characters
            for i, val in enumerate(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']):
                if val in line:
                    digits += str(i + 1)

            digit = str(digits[0]) + str(digits[-1])
            total += int(digit)
            
    print(total)

def day_2():
    p1_total = 0
    p2_total = 0
    red_limit = 12
    green_limit = 13
    blue_limit = 14

    # Create a variable to read files
    filename = "Files/Day_2.txt"

    # Open file and split into "rows" (lines)
    with open(filename) as f:
        lines = f.readlines()
        for game in lines:
            # Split the line into game number and sets
            game_number, sets_str = game.split(":", 1)
            game_number = game_number.replace("Game ", "").strip()
            sets = sets_str.strip().split(";")

            game_valid = True
                    
            red_min, green_min, blue_min = 0, 0, 0

            for set_ in sets:
                set_ = set_.replace("\n", "").replace(":", "")
                colors = set_.split(",")                

                for color in colors:
                    red, green, blue = "0", "0", "0"
                    if "red" in color:
                        red = color.replace("red", "").replace(" ", "")
                    elif "green" in color:
                        green = color.replace("green", "").replace(" ", "")
                    elif "blue" in color:
                        blue = color.replace("blue", "").replace(" ", "")

                    if int(red) > red_limit or int(blue) > blue_limit or int(green) > green_limit:
                        game_valid = False

                    if int(red) > red_min:
                        red_min = int(red)
                    if int(blue) > blue_min:
                        blue_min = int(blue)
                    if int(green) > green_min:
                        green_min = int(green)

            if(red_min == 0):
                red_min = 1
            if(blue_min == 0):
                blue_min = 1
            if(green_min == 0):
                green_min = 1
                

            p2_total += red_min * blue_min * green_min
            if game_valid:
                p1_total += int(game_number)

    print(p1_total)
    print(p2_total)

def day_3():
    board = list(open('Files/Day_3.txt'))
    chars = {(r, c): [] for r in range(140) for c in range(140)
                        if board[r][c] not in '01234566789.'}

    for r, row in enumerate(board):
        for n in re.finditer(r'\d+', row):
            edge = {(r, c) for r in (r-1, r, r+1)
                           for c in range(n.start()-1, n.end()+1)}

            for o in edge & chars.keys():
                chars[o].append(int(n.group()))

    p1 = sum(sum(p)  for p in chars.values()),
    p2 = sum(reduce(lambda x, y: x * y, p) for p in chars.values() if len(p) == 2)

    print(p1, p2)

def day_4():
    p1 = 0
    p2 = 0

    # Create a variable to read files
    filename = "Files/Day_4.txt"
    
    def parse_file(file):
        return [parse_line(ln) for ln in file]

    def parse_line(line):
        game_name, nums = line.split(':')
        win, mine = nums.strip().split('|')
        match = len(set(win.strip().split()) & set(mine.strip().split()))
        return match

    def process_file(file):
        with open(file) as f:
            yield from f.readlines()

    def part1(games):
        return sum(2 ** (n - 1) for n in games if n)

    def part2(games):
        ngames = len(games)
        temp = [None] * ngames
        return sum(
            part2_rec(cn=i, games=games, temp=temp)
            for i in range(0, ngames)
        )

    def part2_rec(cn, games, temp):
        if temp[cn] is not None:
           return temp[cn]
        if cn >= len(games):
            return 0
        total = 1 + sum(
            part2_rec(cn=i, games=games, temp=temp)
            for i in range(cn+1, cn+games[cn]+1)
        )
        temp[cn] = total
        return total    

    input = parse_file(process_file(filename))
    p1 = part1(input)
    p2 = part2(input)

    print (p1, p2)

def day_5():
    p1 = 0
    p2 = 0
    
    # Create a variable to read files
    filename = "Files/Day_4.txt"

    def read_input(filename):
        with open(filename, 'r') as file:
            return file.read().strip()

    def parse_function_string(s):
        lines = s.split('\n')[1:]
        return [[int(x) for x in line.split()] for line in lines]

    class Function:
        def __init__(self, s):
            self.tuples = parse_function_string(s)

        def apply_one(self, x):
            for dst, src, sz in self.tuples:
                if src <= x < src + sz:
                    return x + dst - src
            return x

        def apply_range(self, R):
            A = []
            for dest, src, sz in self.tuples:
                src_end = src + sz
                NR = []
                while R:
                    st, ed = R.pop()
                    before = (st, min(ed, src))
                    inter = (max(st, src), min(src_end, ed))
                    after = (max(src_end, st), ed)
                    if before[1] > before[0]:
                        NR.append(before)
                    if inter[1] > inter[0]:
                        A.append((inter[0] - src + dest, inter[1] - src + dest))
                    if after[1] > after[0]:
                        NR.append(after)
                R = NR
            return A + R

    def apply_functions_to_seed(seed, functions):
        result = []
        for x in seed:
            for f in functions:
                x = f.apply_one(x)
            result.append(x)
        return result

    def apply_functions_to_ranges(ranges, functions):
        result = []
        pairs = list(zip(ranges[::2], ranges[1::2]))
        for st, sz in pairs:
            R = [(st, st + sz)]
            for f in functions:
                R = f.apply_range(R)
            result.append(min(R)[0])
        return result
        
    data = read_input(filename)
    parts = data.split('\n\n')
    seed, *others = parts
    seed = [int(x) if x.isdigit() else 0 for x in seed.split(':')[1].split()]

    functions = [Function(s) for s in others]

    p1 = apply_functions_to_seed(seed, functions)

    p2 = apply_functions_to_ranges(seed, functions)

    print (p1)

def day_5_2():
    filename = "Files\Day_5.txt"
    D = open(filename).read().strip()
    L = D.split('\n')

    parts = D.split('\n\n')
    seed, *others = parts
    seed = [int(x) for x in seed.split(':')[1].split()]

    class Function:
      def __init__(self, S):
        lines = S.split('\n')[1:] # throw away name
        # dst src sz
        self.tuples: list[tuple[int,int,int]] = [[int(x) for x in line.split()] for line in lines]
        #print(self.tuples)
      def apply_one(self, x: int) -> int:
        for (dst, src, sz) in self.tuples:
          if src<=x<src+sz:
            return x+dst-src
        return x

      # list of [start, end) ranges
      def apply_range(self, R):
        A = []
        for (dest, src, sz) in self.tuples:
          src_end = src+sz
          NR = []
          while R:
            # [st                                     ed)
            #          [src       src_end]
            # [BEFORE ][INTER            ][AFTER        )
            (st,ed) = R.pop()
            # (src,sz) might cut (st,ed)
            before = (st,min(ed,src))
            inter = (max(st, src), min(src_end, ed))
            after = (max(src_end, st), ed)
            if before[1]>before[0]:
              NR.append(before)
            if inter[1]>inter[0]:
              A.append((inter[0]-src+dest, inter[1]-src+dest))
            if after[1]>after[0]:
              NR.append(after)
          R = NR
        return A+R

    Fs = [Function(s) for s in others]

    def f(R, o):
      A = []
      for line in o:
        dest,src,sz = [int(x) for x in line.split()]
        src_end = src+sz

    P1 = []
    for x in seed:
      for f in Fs:
        x = f.apply_one(x)
      P1.append(x)
    print(min(P1))

    P2 = []
    pairs = list(zip(seed[::2], seed[1::2]))
    for st, sz in pairs:
      # inclusive on the left, exclusive on the right
      # e.g. [1,3) = [1,2]
      # length of [a,b) = b-a
      # [a,b) + [b,c) = [a,c)
      R = [(st, st+sz)]
      for f in Fs:
        R = f.apply_range(R)
      #print(len(R))
      P2.append(min(R)[0])
    print(min(P2))

def day_6():
    filename = "Files/Day_6.txt"


    def read_lines_to_list() -> List[str]:
        lines: List[str] = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                lines.append(line)

        return lines


    def calculate_distance(held, remaining):
        return remaining * held


    def part_one():
        lines = read_lines_to_list()
        times = [int(val) for val in lines[0].split(":")[1].strip().split()]
        records = [int(val) for val in lines[1].split(":")[1].strip().split()]

        races = [(time, distance) for (time, distance) in zip(times, records)]

        num_ways = []

        for time, record in races:
            wins = 0
            for i in range(time + 1):
                if calculate_distance(i, time - i) > record:
                    wins += 1

            num_ways.append(wins)

        result = reduce((lambda x, y: x * y), num_ways)
        print(f"Part 1: {result}")


    def part_two():
        lines = read_lines_to_list()
        time = int("".join(lines[0].split(":")[1].strip().split()))
        record = int("".join(lines[1].split(":")[1].strip().split()))

        wins = 0
        for i in range(time + 1):
            if calculate_distance(i, time - i) > record:
                wins += 1

        print(f"Part 1: {wins}")


    part_one()
    part_two()

def day_7():
    filename = "Files/Day_7.txt"
    def eval(line):
        hand, bid = line.split()
        hand = hand.translate(str.maketrans('TJQKA', face))
        best = max(type(hand.replace('0', r)) for r in hand)
        return best, hand, int(bid)

    def type(hand):
        return sorted(map(hand.count, hand), reverse=True)

    for face in 'ABCDE', 'A0CDE':
        print(sum(rank * bid for rank, (*_, bid) in
            enumerate(sorted(map(eval, open(filename))), start=1)))

def day_8():
    filename = "Files/Day_8.txt"

    def pt_1():
        START = 'AAA'
        GOAL = 'ZZZ'
    
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        seq, net = text.split('\n\n')
        net = ''.join(filter(lambda x: x not in list('=(,)'), list(net))).strip()
        net = [x.split() for x in net.split('\n')]
        nodes = {x[0]: (x[1], x[2]) for x in net}

        node = START
        idx = 0
        steps = 0
        while node != GOAL:
            dir = 0 if seq[idx] == 'L' else 1
            node = nodes[node][dir]
            idx = idx + 1 if idx < len(seq) - 1 else 0
            steps += 1

        print(steps)

    def pt_2():
        START = 'A'
        GOAL = 'Z'

        def lcm(list_int):
            result = 1
            for i in len(list_int):
                result *= result * i
            return result

        def count_steps(nodes, start):
            node = start
            idx = 0
            steps = 0
            while node[2] != GOAL:
                dir = 0 if seq[idx] == 'L' else 1
                node = nodes[node][dir]
                idx = idx + 1 if idx < len(seq) - 1 else 0
                steps += 1
            return steps

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return a * b // gcd(a, b)

        def lcm_of_list(numbers):
            result = 1
            for num in numbers:
                result = lcm(result, num)
            return result

        if __name__ == "__main__":
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            # parse
            seq, net = text.split('\n\n')
            net = ''.join(filter(lambda x: x not in list('=(,)'), list(net))).strip()
            net = [x.split() for x in net.split('\n')]
            nodes = {x[0]: (x[1], x[2]) for x in net}
            # find steps for each path
            startnodes = [x for x in list(nodes.keys()) if x[2] == START]
            steps = [count_steps(nodes, x) for x in startnodes]
            # compute when their cycles meet
            print(lcm_of_list(steps))

    pt_1()
    pt_2()

if __name__ == '__main__':
    day_8()