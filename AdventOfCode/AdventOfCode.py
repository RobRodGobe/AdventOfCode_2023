import sys
import re
from functools import reduce

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


if __name__ == '__main__':
    day_3()