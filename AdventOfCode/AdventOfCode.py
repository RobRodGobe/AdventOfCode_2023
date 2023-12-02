import sys
import re

def main():
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

if __name__ == '__main__':
    main()