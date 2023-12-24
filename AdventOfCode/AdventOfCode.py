from gettext import find
from msilib.schema import File
import queue
import sys
import re
from functools import reduce, lru_cache
from collections import defaultdict, deque
from typing import List, Tuple, Dict
import math, time
import heapq
from functools import reduce
import operator
from heapq import heapify, heappush, heappop
import numpy as np
from sympy import Symbol
from sympy import solve_poly_system

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

def day_9():
    filename = "Files/Day_9.txt"

    file = open(filename).read().strip()
    splits = file.split('\n')

    p1 = 0
    p2 = 0

    def calculate_differences(numbers):
        differences = [int(numbers[i + 1]) - int(numbers[i]) for i in range(len(numbers) - 1)]
        return differences

    def calculate_sum(numbers):
        return sum(numbers)

    def sum_last_elements(array):
        l = len(array)    
        if (l == 1):
            return array[0]
        array[l - 2] += array[l - 1]
        del(array[l - 1])
        return sum_last_elements(array)

    def get_last_element(line, last_elements):
        sum_line = calculate_sum(line)
        last_element = line[len(line) - 1]
        last_elements.append(last_element)
        if sum_line == 0:
            return sum_last_elements(last_elements)
        new_line = calculate_differences(line)
        return get_last_element(new_line, last_elements)

    def sum_first_elements(array):
        l = len(array)    
        if (l == 1):
            return array[0]
        array[l - 2] -= array[l - 1]
        del(array[l - 1])
        return sum_first_elements(array)

    def get_first_element(line, first_elements):
        sum_line = calculate_sum(line)
        first_element = line[0]
        first_elements.append(first_element)
        if sum_line == 0:
            return sum_first_elements(first_elements)
        new_line = calculate_differences(line)
        return get_first_element(new_line, first_elements)

    for line in splits:
        int_line = list(map(int, line.split(' ')))    
        last_elements = []
        first_elements = []
        p1 += get_last_element(int_line, last_elements)
        p2 += get_first_element(int_line, first_elements)


    print (p1)
    print (p2)

def day_10():
    p1 = 0
    p2 = 0
    filename = "Files/Day_10.txt"

    with open(filename, "r") as file:
        grid = [list(line) for line in file.read().split("\n")]

    for row in range(len(grid)):
        for column in range(len(grid[row])):
            if grid[row][column] == 'S':
                starting_row = row
                starting_column = column
                break
        else:
            continue
        break

    check_pipes = [(starting_row, starting_column)]
    seen_pipes = {(starting_row, starting_column)}

    while len(check_pipes) > 0:
        row, column = check_pipes.pop(0)
        current_pipe = grid[row][column]

        if row > 0 and current_pipe in "S|LJ" and grid[row - 1][column] in "|7F" and (row - 1, column) not in seen_pipes:
            seen_pipes.add((row - 1, column))
            check_pipes.append((row - 1, column))

        if row < len(grid) - 1 and current_pipe in "S|7F" and grid[row + 1][column] in "|LJ" and (row + 1, column) not in seen_pipes:
            seen_pipes.add((row + 1, column))
            check_pipes.append((row + 1, column))

        if column > 0 and current_pipe in "S-7J" and grid[row][column - 1] in "-LF" and (row, column - 1) not in seen_pipes:
            seen_pipes.add((row, column - 1))
            check_pipes.append((row, column - 1))

        if column < len(grid[row]) - 1 and current_pipe in "S-LF" and grid[row][column + 1] in "-J7" and (row, column + 1) not in seen_pipes:
            seen_pipes.add((row, column + 1))
            check_pipes.append((row, column + 1))

    p1 = len(seen_pipes) // 2

    
    for row in range(len(grid)):
        for column in range(len(grid[row])):
            if grid[row][column] == 'S':
                starting_row = row
                starting_column = column
                break
        else:
            continue
        break

    check_pipes = [(starting_row, starting_column)]
    seen_pipes = {(starting_row, starting_column)}
    potential_s = {'|', '-', 'L', 'J', '7', 'F'}

    while len(check_pipes) > 0:
        row, column = check_pipes.pop(0)
        current_pipe = grid[row][column]

        if row > 0 and current_pipe in "S|LJ" and grid[row - 1][column] in "|7F" and (row - 1, column) not in seen_pipes:
            seen_pipes.add((row - 1, column))
            check_pipes.append((row - 1, column))
            if current_pipe == 'S':
                potential_s = potential_s.intersection({'|', 'L', 'J'})

        if row < len(grid) - 1 and current_pipe in "S|7F" and grid[row + 1][column] in "|LJ" and (row + 1, column) not in seen_pipes:
            seen_pipes.add((row + 1, column))
            check_pipes.append((row + 1, column))
            if current_pipe == 'S':
                potential_s = potential_s.intersection({'|', '7', 'F'})

        if column > 0 and current_pipe in "S-7J" and grid[row][column - 1] in "-LF" and (row, column - 1) not in seen_pipes:
            seen_pipes.add((row, column - 1))
            check_pipes.append((row, column - 1))
            if current_pipe == 'S':
                potential_s = potential_s.intersection({'-', '7', 'J'})

        if column < len(grid[row]) - 1 and current_pipe in "S-LF" and grid[row][column + 1] in "-J7" and (row, column + 1) not in seen_pipes:
            seen_pipes.add((row, column + 1))
            check_pipes.append((row, column + 1))
            if current_pipe == 'S':
                potential_s = potential_s.intersection({'-', 'L', 'F'})

    s_pipe = potential_s.pop()
    grid[starting_row][starting_column] = s_pipe

    grid = [['.' if (row, column) not in seen_pipes else grid[row][column]
             for column in range(len(grid[row]))] for row in range(len(grid))]


    p2 = 0
    for row in grid:
        for i, char in enumerate(row):
            if char != ".":
                continue

            intersect = 0
            corner_pipes = []
            for j in range(i + 1, len(row)):
                if row[j] in "|":
                    intersect += 1
                if row[j] in "FL":
                    corner_pipes.append(row[j])
                if len(corner_pipes) != 0 and row[j] == "J" and corner_pipes[-1] == "F" or row[j] == "7" and corner_pipes[-1] == "L":
                    corner_pipes.pop(-1)
                    intersect += 1

            if intersect % 2 == 1:
                p2 += 1

    print (p1)
    print (p2)

def day_11():
    filename = "Files/Day_11.txt"
    p1 = 0
    p2 = 0
    xs, ys = zip(*[(x,y) for y,r in enumerate(open(filename))
                     for x,c in enumerate(r) if c == '#'])
    
    def dist(ps):
        ps = [sum((l, 1)[p in ps] for p in range(p)) for p in ps]
        return sum(abs(a-b) for a in ps for b in ps)//2


    for l in 2, 1_000_000:
        p1, p2 = [sum(map(dist, [xs, ys])) for l in (2, 1_000_000)]

        
    print (p1, p2)

def day_12():
    filename = "Files/Day_12.txt"
    p1 = 0
    p2 = 0
    

    @lru_cache(maxsize=128, typed=False) # too slow without the cache - stpres function values
    def numlegal(s,c):

        s = s.lstrip('.') # ignore leading dots

        # ['', ()] is legal
        if s == '':
            return int(c == ()) 

        # [s, ()] is legal so long as s has no '#' (we can convert '?' to '.')
        if c == ():
            return int(s.find('#') == -1) 

        # s starts with '#' so remove the first spring
        if s[0] == '#':
            if len(s) < c[0] or '.' in s[:c[0]]:
                return 0 # impossible - not enough space for the spring
            elif len(s) == c[0]:
                return int(len(c) == 1) #single spring, right size
            elif s[c[0]] == '#':
                return 0 # springs must be separated by '.' (or '?') 
            else:
                return numlegal(s[c[0]+1:],c[1:]) # one less spring

        # numlegal springs if we convert the first '?' to '#' + '.'
        return numlegal('#'+s[1:],c) + numlegal(s[1:],c)


    springs = [c.strip().split() for c in open(filename).readlines()]
    ss = [[c[0],tuple(int(d) for d in c[1].split(','))] for c in springs]
    p1 = sum(numlegal(s,c) for [s,c] in ss)


    ss2 = [[(s[0]+'?')*4 + s[0],s[1]*5] for s in ss]
    p2 = sum(numlegal(s,c) for [s,c] in ss2)


        
    print (p1, p2)

def day_13():
    filename = "Files/Day_13.txt"

    p1 = 0
    p2 = 0


    def parse_input(data):
        # Split each map and then return that list
        data_list = data.split("\n\n")
        return data_list

    def split_horizontal_vertical(ash_map):
        # Return a list of both rows and columns from the passed map
        ash_map_split = ash_map.splitlines()
        return ash_map_split, list(zip(*ash_map_split))

    def find_reflect_point(ash_map, part2=False):
        # Run through each line of the passed map and the lines match spread out and find matching lines until you hit the end of the list
        # If this is part2, introduce smudge function and return x value if a reflection is found
        for idx, line in enumerate(ash_map):
            try:  # Use a try except to avoid tracking length
                smudge = False if not part2 else find_smudge(line, ash_map[idx + 1])  # Return a bool from find_smudge only if part2 is True
                if ash_map[idx + 1] == line or smudge:  # If the line matches, or if it's part2 and smudge matches, spread out and check for matches
                    reflected = True
                    close_end = (idx + 1) if (len(ash_map) // 2) > idx else len(ash_map) - (idx + 1)  # Find the length to the closest end of the list
                    for i in range(1, close_end):
                        if ash_map[idx + i + 1] != ash_map[idx - i]:  # If the lines don't match, reflection is False unless it's part2
                            if part2 and not smudge and find_smudge(ash_map[idx + i + 1], ash_map[idx - i]):  # If it's part2, and we have not found a smudge yet, check for smudge
                                smudge = True
                                continue
                            else:
                                reflected = False
                    if part2:
                        if reflected and smudge:  # Part2 only counts if a smudge has been found
                            return (idx + 1)
                    elif reflected:
                        return (idx + 1)
            except IndexError as e:  # Continue on if you found an index error
                pass
        return

    def find_smudge(line1, line2):
        # If there's only one character different between the lists, return True else False
        return sum(line1[i] != line2[i] for i in range(len(line1))) == 1

    def part1(parsed_data):
        # Go through each map, split the maps into horizontal and vertical, and process starting with horizontal first
        # If reflected on the horizontal, multiply by 100 before adding to total
        count = 1
        total = 0
        for ash_map in parsed_data:
            row_map, column_map = split_horizontal_vertical(ash_map)
            horizontal_reflect = find_reflect_point(row_map)
            if horizontal_reflect is None:
                vertical_reflect = find_reflect_point(column_map)
                count += 1
                total += vertical_reflect
            else:
                count += 1
                total += (horizontal_reflect * 100)

        return total

    def part2(parsed_data):
        # Go through each map, split the maps into horizontal and vertical, and process starting with horizontal first using the smudge mechanic
        # If reflected on the horizontal, multiply by 100 before adding to total
        count = 1
        total = 0
        for ash_map in parsed_data:
            row_map, column_map = split_horizontal_vertical(ash_map)
            horizontal_reflect = find_reflect_point(row_map, True)
            if horizontal_reflect is None:
                vertical_reflect = find_reflect_point(column_map, True)
                count += 1
                total += vertical_reflect
            else:
                count += 1
                total += (horizontal_reflect * 100)
        return total

    with open(filename , "r") as f:
            parsed_data = parse_input(f.read())
            
    p1 = part1(parsed_data)
    
    p2 = part2(parsed_data)

    print (p1, p2)

def day_14():
    filename = "Files/Day_14.txt"
    p1 = 0
    p2 = 0

    with open(filename, "r") as file:
        platform = file.read().splitlines()

    platform = [list(row[::-1]) for row in zip(*platform)]

    for row in platform:
        for i, char in reversed(list(enumerate(row))):
            if char == "O":
                previous_index = i
                current_index = i
                reached_end = False
                while not reached_end:
                    if current_index + 1 < len(row) and row[current_index + 1] not in "#O":
                        current_index += 1
                    else:
                        reached_end = True
                row[previous_index] = "."
                row[current_index] = "O"

    platform = [[platform[j][i] for j in range(
        len(platform))] for i in range(len(platform[0])-1, -1, -1)]

    p1 = 0
    current_row_num = len(platform)
    for row in platform:
        for char in row:
            if char == "O":
                p1 += current_row_num
        current_row_num -= 1
        
    def tilt_north(platform):
        platform = [list(row[::-1]) for row in zip(*platform)]

        for row in platform:
            for i, char in reversed(list(enumerate(row))):
                if char == "O":
                    previous_index = i
                    current_index = i
                    reached_end = False
                    while not reached_end:
                        if current_index + 1 < len(row) and row[current_index + 1] not in "#O":
                            current_index += 1
                        else:
                            reached_end = True
                    row[previous_index] = "."
                    row[current_index] = "O"

        return platform


    def calculate_north_wall_load(platform):
        north_wall_load = 0
        current_row_num = len(platform)
        for row in platform:
            for char in row:
                if char == "O":
                    north_wall_load += current_row_num
            current_row_num -= 1

        return north_wall_load


    with open(filename, "r") as file:
        platform = file.read().splitlines()

    seen_list = []
    cycle_loop = []

    for current_cycle in range(1, 1000000001):
        for _ in range(4):
            platform = tilt_north(platform)

        if platform in seen_list:
            index = seen_list.index(platform)
            cycle_loop = seen_list[index:]
            remaining_cycles = 1000000000 - current_cycle
            index_billionth = remaining_cycles % len(cycle_loop)
            p2 = calculate_north_wall_load(cycle_loop[index_billionth])
            break

        seen_list.append(platform)

    print (p1, p2)
    
def day_15():
    filename = "Files/Day_15.txt"
    p1 = 0
    p2 = 0

    with open(filename, "r") as file:
        sequence: List[str] = file.read().split(',')


    def _hash(_s: str):
        h: int = 0
        for i in _s:
            h = (h + ord(i)) * 17 % 256
        return h


    p1 = sum(_hash(i) for i in sequence)


    # Part 2
    boxes: Dict[int, Dict[str, int]] = {i: {} for i in range(256)}

    for op in sequence:
        if op[-2] == '=':
            s, v = op.split('=')
            boxes[_hash(s)][s] = int(v)
        elif op[-1] == '-':
            s: str = op[:-1]
            hsh: int = _hash(s)
            if s in boxes[hsh]:
                del boxes[hsh][s]

    p2 = sum(sum((m + 1) * v * (i + 1) for i, v in enumerate(box.values())) for m, box in enumerate(boxes.values()))

    print (p1, p2)

    
def day_16():
    filename = "Files/Day_16.txt"
    p1 = 0
    p2 = 0

    with open(filename, "r") as file:
        puzzle_input = file.read()

    def part1(puzzle_input):

        grid = [list(r) for r in puzzle_input.split('\n')]
        m, n = len(grid), len(grid[0])
        visited = set()
        energized = set()
        queue = set([(0, 0, 'right')])   
        while queue:
            x, y, direction = queue.pop()
            energized.add((x, y))
            tile = grid[x][y]

            if y < n-1 and (x, y+1, 'right') not in visited and (
                    (direction == 'right' and tile in '.-') or 
                    (direction == 'up' and tile in '/-') or
                    (direction == 'down' and tile in '\\-')):
                queue.add((x, y+1, 'right'))
                visited.add((x, y+1, 'right'))

            if x > 0 and (x-1, y, 'up') not in visited and (
                    (direction == 'up' and tile in '.|') or 
                    (direction == 'right' and tile in '/|') or
                    (direction == 'left' and tile in '\\|')):
                queue.add((x-1, y, 'up'))
                visited.add((x-1, y, 'up'))

            if y > 0 and (x, y-1, 'left') not in visited and (
                    (direction == 'left' and tile in '.-') or 
                    (direction == 'up' and tile in '\\-') or
                    (direction == 'down' and tile in '/-')):
                queue.add((x, y-1, 'left'))
                visited.add((x, y-1, 'left'))

            if x < m-1 and (x+1, y, 'down') not in visited and (
                    (direction == 'down' and tile in '.|') or 
                    (direction == 'right' and tile in '\\|') or
                    (direction == 'left' and tile in '/|')):
                queue.add((x+1, y, 'down'))     
                visited.add((x+1, y, 'down'))

        return len(energized)


    def part2(puzzle_input):
    
        grid = [list(r) for r in puzzle_input.split('\n')]
        m, n = len(grid), len(grid[0])
        initial = ({(x, 0, 'right') for x in range(m)} |
                   {(x, n-1, 'left') for x in range(m)} |
                   {(m-1, y, 'up') for y in range(n)} |
                   {(0, y, 'down') for y in range(n)})
    
        best = 0
        for i in initial:
            visited = set()
            energized = set()
            queue = set([i])   
            while queue:
                x, y, direction = queue.pop()
                energized.add((x, y))
                tile = grid[x][y]

                if y < n-1 and (x, y+1, 'right') not in visited and (
                        (direction == 'right' and tile in '.-') or 
                        (direction == 'up' and tile in '/-') or
                        (direction == 'down' and tile in '\\-')):
                    queue.add((x, y+1, 'right'))
                    visited.add((x, y+1, 'right'))

                if x > 0 and (x-1, y, 'up') not in visited and (
                        (direction == 'up' and tile in '.|') or 
                        (direction == 'right' and tile in '/|') or
                        (direction == 'left' and tile in '\\|')):
                    queue.add((x-1, y, 'up'))
                    visited.add((x-1, y, 'up'))

                if y > 0 and (x, y-1, 'left') not in visited and (
                        (direction == 'left' and tile in '.-') or 
                        (direction == 'up' and tile in '\\-') or
                        (direction == 'down' and tile in '/-')):
                    queue.add((x, y-1, 'left'))
                    visited.add((x, y-1, 'left'))

                if x < m-1 and (x+1, y, 'down') not in visited and (
                        (direction == 'down' and tile in '.|') or 
                        (direction == 'right' and tile in '\\|') or
                        (direction == 'left' and tile in '/|')):
                    queue.add((x+1, y, 'down'))     
                    visited.add((x+1, y, 'down'))

            best = max(best, len(energized))

        return best


    p1 = part1(puzzle_input)
    p2 = part2(puzzle_input)

    print (p1, p2)

def day_17():
    filename = "Files/Day_17.txt"
    p1 = 0
    p2 = 0

    def minimal_heat(start, end, least, most):
        queue = [(0, *start, 0,0)]
        seen = set()
        while queue:
            heat,x,y,px,py = heapq.heappop(queue)
            if (x,y) == end: return heat
            if (x,y, px,py) in seen: continue
            seen.add((x,y, px,py))
            # calculate turns only
            for dx,dy in {(1,0),(0,1),(-1,0),(0,-1)}-{(px,py),(-px,-py)}:
                a,b,h = x,y,heat
                # enter 4-10 moves in the chosen direction
                for i in range(1,most+1):
                    a,b=a+dx,b+dy
                    if (a,b) in board:
                        h += board[a,b]
                        if i>=least:
                            heapq.heappush(queue, (h, a,b, dx,dy))

    board = {(i,j): int(c) for i,r in enumerate(open(filename)) for j,c in enumerate(r.strip())}
    p1 = minimal_heat((0,0),max(board), 1, 3)
    p2 = minimal_heat((0,0),max(board), 4, 10)

    print (p1, p2)

def day_18():
    filename = "Files/Day_18.txt"
    p1 = 0
    p2 = 0


    mm = list(map(lambda x: x.strip().split(' '), open(filename, "r").readlines()))
    dd = {'R': (0, 1), 'D': (1, 0), 'L': (0, -1), 'U': (-1, 0)}  

    def f(mm):
        p = []
        x = 0
        y = 0
        for m in mm:
            (d, l) = m
            if d == 'D':
                p.append((y, x, 'D'))
                p.append((y + l, x, 'U'))
            elif d == 'U': 
                p.append((y, x, 'U'))
                p.append((y - l, x, 'D'))
            y = y + (dd[d][0] * l)
            x = x + (dd[d][1] * l)
        p.sort(key=(lambda x: (1000 if x[2] == 'U' else 0) + x[0] * 1000000 + x[1]))

        l = []
        q = 0
        t = 0
        while q < len(p):
            ny = p[q][0]
            i = 0
            while i < len(l):
                if ny > y + 1 and l[i+1] > l[i]:
                    t += ((l[i+1] - l[i])+ 1) * (ny - y - 1)
                i += 2
            y = ny
            a = l.copy()
            while q < len(p) and p[q][0] == ny:
                if p[q][2] == 'D':
                    l.append(p[q][1])
                else:
                    l.remove(p[q][1])
                q += 1
            l.sort()
            b = l.copy()
            i = j = 0
            tt = t
            while (i < len(a) or j < len(b)):
                if i < len(a) and j < len(b):
                    if a[i] > b[j+1]:
                        t += ((b[j+1] - b[j]) + 1)
                        j += 2
                    elif b[j] > a[i+1]:
                        t += ((a[i+1] - a[i]) + 1)
                        i += 2
                    elif a[i+1] > b[j+1]:
                        a[i] = min(a[i], b[j])
                        j += 2
                    else:
                        b[j] = min(a[i], b[j])
                        i += 2
                elif i < len(a):
                    t += ((a[i+1] - a[i]) + 1)
                    i += 2
                else:
                    t += ((b[j+1] - b[j]) + 1)
                    j += 2
        return t

    p1 = f(list([(m[0], int(m[1])) for m in mm]))
    DD = { 0: 'R', 1: 'D', 2: 'L', 3: 'U' }
    p2 = f(list([(DD[int(m[2][7])], int(m[2][2:7], 16)) for m in mm]))

    print (p1, p2)

def day_19():
    filename = "Files/Day_19.txt"
    p1 = 0
    p2 = 0

    rule_pat = re.compile(r'([a-z]+)(.)(\d+):(.+)')
    def parse_input(path):
        workflows = {}
        parts = []

        for line in open(path).read().strip().split('\n'):
            if ':' in line:
                workflow = {'rules': []}
                workflow['name'], rules_s = line.rstrip('}').split('{')
                rules = []
                for rule_s in rules_s.split(','):
                    if ':' in rule_s:
                        key, op, val, target = rule_pat.search(rule_s).groups()
                        workflow['rules'].append(((key, op, int(val)), target))
                    else:
                        workflow['rules'].append((None, rule_s))
                workflows[workflow['name']] = workflow
            elif line:
                part = {}
                for word in line.strip('{}\n').split(','):
                    k, v = word.split('=')
                    part[k] = int(v)
                parts.append(part)
        return workflows, parts


    def accept_part(part, workflows):
        workflow = workflows['in']
        while True:
            for cond, target in workflow['rules']:
                if cond is None:
                    result = target
                else:
                    result = None
                    key, op, val = cond
                    if op == '<' and part[key] < val:
                        result = target
                    elif op == '>' and part[key] > val:
                        result = target
                if result == 'A':
                    return True
                elif result == 'R':
                    return False
                elif result:
                    workflow = workflows[result]
                    break
                else:
                    assert result is None


    def add_constraint(constraints, condition):
        key, op, val = condition
        lo, hi = constraints.get(key, (1, 4000))
        if op == '>':
            if val >= hi:
                return None
            lo = val + 1
        else:
            if val <= lo:
                return None
            hi = val - 1
        return dict(constraints, **{key: (lo, hi)})


    def invert(condition):
        key, op, val = condition
        return (key, '>', val - 1) if (op == '<') else (key, '<', val + 1)


    def trace_paths(workflows, state):
        workflow_name, constraints = state
        for condition, target in workflows[workflow_name]['rules']:
            if condition is None:
                cons_true = constraints
            else:
                cons_true = add_constraint(constraints, condition)
                constraints = add_constraint(constraints, invert(condition))
            if cons_true is not None:
                if target == 'A':
                    yield cons_true
                elif target != 'R':
                    yield from trace_paths(workflows, (target, cons_true))


    def count_paths(paths):
        total = 0
        for path in paths:
            total += reduce(int.__mul__, [hi - lo + 1 for lo, hi in path.values()])
        return total

    workflows, parts = parse_input(filename)
    p1 = sum(sum(p.values()) for p in parts if accept_part(p, workflows))
    initial_state = ('in', {r: (1, 4000) for r in 'xmas'})
    p2 = count_paths(trace_paths(workflows, initial_state))
    
    
    print (p1, p2)

def day_20():
    filename = "Files/Day_20.txt"
    p1 = 0
    p2 = 0
    
    def load(file):
      with open(file) as f:
        return [row.strip().split(' -> ') for row in f]
  
    class Module():
      def __init__(self,name,type,dest):
        self.name = name
        self.type = type
        self.dest = dest
        if type == '%': 
            self.mem = False
        elif type == '&': 
            self.mem = {}
        else:
            self.mem = None  

      def __repr__(self):
        return f'Name: {self.name} Type: {self.type} Dest: {self.dest} Mem: {self.mem}'
  
      def receive_impulse(self,impulse,last):
        if self.type == '%':
          self.mem = not self.mem
          return self.mem
    
        if self.type == '&':
          self.mem[last] = impulse
          return not all(self.mem.values())

    
    def solve(p):
      modules = dict()
      for module, destinations in p:
        curr = [d.strip() for d in destinations.split(',')]
        if module == 'broadcaster':
          modules[module] = Module('broadcaster',None,curr)
        else:
          modules[module[1:]] = Module(module[1:],module[0],curr)
  
      for object in modules.values():
        for dest in object.dest:
          if dest not in modules: continue
          obj2 = modules[dest]
          if obj2.type != '&': continue
          obj2.mem[object.name]=False
  
      main_module = [m.name for m in modules.values() if 'rx' in m.dest][0]
      lowHigh, cycles  = [0,0], {m:0 for m in modules[main_module].mem}
  
      for buttons in range(1,10_000): 
        if all(cycles.values()): break
        queue = [(dest,False,'broadcaster') for dest in modules['broadcaster'].dest]
        if buttons < 1001: lowHigh[0] += 1
    
        while queue:
          curr, impulse, last = queue.pop(0)
          if buttons < 1001: lowHigh[impulse] += 1
          if curr not in modules: continue
          curr = modules[curr]

          if curr.name == main_module and impulse:
            cycles[last] = buttons - cycles[last]
            
          if curr.type == '%' and impulse: continue
          impulse = curr.receive_impulse(impulse,last)
      
          for nxt in curr.dest:
            queue.append((nxt, impulse, curr.name))
            
      return lowHigh[0] * lowHigh[1], lcm(*list(cycles.values()))

    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    def lcm(*args):
        if not args:
            return 1
        result = args[0]
        for value in args[1:]:
            result = result * value // gcd(result, value)
        return result

    
    resolved = solve(load(filename))
    p1 = resolved[0]
    p2 = resolved[1]

    print (p1, p2)

def day_21():
    filename = "Files/Day_21.txt"
    p1 = 0
    p2 = 0
    
    p1_steps = 64
    p2_steps = 26501365

    with open(filename, "r") as file:
        data = file.read().splitlines()
        ln, steps, cycle, seen, even, odd, n = len(data), 0, [], set(), set(), set(), p2_steps
        grid = {(x, y): data[y][x] for x in range(ln) for y in range(ln)}
        queue = [(steps, next((k for k, v in grid.items() if v == "S")))]
        heapify(queue)
        while queue:
            new_steps, (x, y) = heappop(queue)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            if new_steps != steps:
                if steps == p1_steps:
                    p1 = len(even)
                if steps % (ln * 2) == n % (ln * 2):
                    if len(cycle) == 3:
                        p2, offset, increment = cycle[0], cycle[1] - cycle[0], (
                                    cycle[2] - cycle[1]) - (cycle[1] - cycle[0])
                        for x in range(n // (ln * 2)):
                            p2 += offset
                            offset += increment
                        break
                    cycle.append(len([even, odd][steps % 2]))
            steps, next_steps = new_steps, new_steps + 1
            for a, b in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if grid[a % ln, b % ln] != "#":
                    if not next_steps % 2:
                        even.add((a, b))
                    else:
                        odd.add((a, b))
                    heappush(queue, (next_steps, (a, b)))
    
    print (p1, p2)

def day_22():
    filename = "Files/Day_22.txt"
    p1 = 0
    p2 = 0
    
    brick = []
    file = open(filename, "r").readlines()
    for line in file:
        a,b = line.split("~")
        a = list(map(int, a.split(",")))
        b = list(map(int, b.split(",")))
        brick.append((a,b))

    n = len(brick)
    

    brick.sort(key=lambda x: x[0][2])

    highest = defaultdict(lambda:(0,-1))
    bad = set()
    graph = [[] for i in range(n)]
    for idx,b in enumerate(brick):
        mxh = -1
        support_set = set()
        for x in range(b[0][0], b[1][0]+1):
            for y in range(b[0][1], b[1][1]+1):
                if highest[x,y][0] + 1 > mxh:
                    mxh = highest[x,y][0] + 1
                    support_set = {highest[x,y][1]}
                elif highest[x,y][0] + 1 == mxh:
                    support_set.add(highest[x,y][1])
    
        for x in support_set:
            if x != -1:
                graph[x].append(idx)

        if len(support_set) == 1:
            bad.add(support_set.pop())
    
        fall = b[0][2] - mxh
        if fall > 0:
            b[0][2] -= fall
            b[1][2] -= fall

        for x in range(b[0][0], b[1][0]+1):
            for y in range(b[0][1], b[1][1]+1):
                highest[x,y] = (b[1][2], idx)

    p1 = n - len(bad) + 1

    def count(idx, graph):
        indeg = [0 for __ in range(n)]
        for j in range(n):
            for i in graph[j]:
                indeg[i] += 1
        q = [idx]
        count = -1
        while len(q) > 0:
            count += 1
            x = q.pop()
            for i in graph[x]:
                indeg[i] -= 1
                if indeg[i] == 0:
                    q.append(i)

        return count

    p2 = sum(count(x, graph) for x in range(n))

    
    print (p1, p2)

def day_23(filename="Files/Day_23.txt"):
    p1, p2 = 0, 0

    def neighbors(grid, r, c, ignore_slopes):
        cell = grid[r][c]

        if ignore_slopes or cell == ".":
            for r, c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if grid[r][c] != "#":
                    yield r, c
        elif cell == "v":
            yield r + 1, c
        elif cell == "^":
            yield r - 1, c
        elif cell == ">":
            yield r, c + 1
        elif cell == "<":
            yield r, c - 1

    def num_neighbors(grid, r, c, ignore_slopes):
        if ignore_slopes or grid[r][c] == ".":
            return sum(
                grid[r][c] != "#" for r, c in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
            )
        return 1

    def is_node(grid, rc, src, dst, ignore_slopes):
        return rc == src or rc == dst or num_neighbors(grid, *rc, ignore_slopes) > 2

    def adjacent_nodes(grid, rc, src, dst, ignore_slopes):
        q = deque([(rc, 0)])
        seen = set()

        while q:
            rc, dist = q.popleft()
            seen.add(rc)

            for n in neighbors(grid, *rc, ignore_slopes):
                if n in seen:
                    continue

                if is_node(grid, n, src, dst, ignore_slopes):
                    yield n, dist + 1
                    continue

                q.append((n, dist + 1))

    def graph_from_grid(grid, src, dst, ignore_slopes=False):
        g = defaultdict(list)
        q = deque([src])
        seen = set()

        while q:
            rc = q.popleft()
            if rc in seen:
                continue

            seen.add(rc)

            for n, weight in adjacent_nodes(grid, rc, src, dst, ignore_slopes):
                g[rc].append((n, weight))
                q.append(n)

        return g

    def longest_path(g, cur, dst, distance=0, seen=set()):
        if cur == dst:
            return distance

        best = 0
        seen.add(cur)

        for neighbor, weight in g[cur]:
            if neighbor in seen:
                continue

            best = max(best, longest_path(g, neighbor, dst, distance + weight))

        seen.remove(cur)
        return best

    with open(filename, "r") as fin:
        grid = list(map(list, fin.read().splitlines()))
        height, width = len(grid), len(grid[0])

        grid[0][1] = "#"
        grid[height - 1][width - 2] = "#"

        src = (1, 1)
        dst = (height - 2, width - 2)

        g = graph_from_grid(grid, src, dst)
        pathlen = longest_path(g, src, dst) + 2
        p1 = pathlen

        g = graph_from_grid(grid, src, dst, ignore_slopes=True)
        pathlen = longest_path(g, src, dst) + 2
        p2 = pathlen

    print(p1, p2)

def day_24(filename="Files/Day_24.txt"):
    p1, p2 = 0, 0

    handle = open(filename,"r")
    
    shards = []
    for line in handle:
      pos, vel = line.strip().split(" @ ")
      px,py,pz = pos.split(", ")
      vx,vy,vz = vel.split(", ")
      shards.append((int(px),int(py),int(pz),int(vx),int(vy),int(vz)))

    count = 0

    for adx in range(len(shards)-1):
      shard_a = shards[adx]
      ma = shard_a[4]/shard_a[3] # pendiente de la recta
      ba = shard_a[1] - ma * shard_a[0] # ordenada a la base
      for bdx in range(adx+1,len(shards)):
        shard_b = shards[bdx]
        mb = shard_b[4]/shard_b[3] # pendiente de la recta
        bb = shard_b[1] -mb * shard_b[0] # ordenada a la base
        if ma == mb: # parallel lines
          if ba == bb: # sanity check
            print(shard_a,shard_b,"this is the same picture") # silly reference to The Office
            exit()
          continue
        ix = (bb - ba)/(ma - mb)
        iy = ma*ix + ba

        ta = (ix - shard_a[0])/shard_a[3]
        tb = (ix - shard_b[0])/shard_b[3]

        if ta >= 0 and tb >= 0 and ix >= 200000000000000 and ix <= 400000000000000 and iy >= 200000000000000 and iy <= 400000000000000:
          count+=1

    p1 = count
    
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    vx = Symbol('vx')
    vy = Symbol('vy')
    vz = Symbol('vz')

    equations = []
    t_syms = []
    for idx,shard in enumerate(shards[:3]):
      x0,y0,z0,xv,yv,zv = shard
      t = Symbol('t'+str(idx)) # represents each intersection

      eqx = x + vx*t - x0 - xv*t
      eqy = y + vy*t - y0 - yv*t
      eqz = z + vz*t - z0 - zv*t

      equations.append(eqx)
      equations.append(eqy)
      equations.append(eqz)
      t_syms.append(t)
      
    result = solve_poly_system(equations,*([x,y,z,vx,vy,vz]+t_syms))
    p2 = result[0][0]+result[0][1]+result[0][2]

    print(p1, p2)

if __name__ == "__main__":
    day_24()