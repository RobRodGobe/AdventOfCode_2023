import sys

# Main method
def main():
    # Create a variable to read files
    filename = "Files/Day_1.txt"
    total = 0

    # Open file and split into "rows" (lines)
    with open(filename) as f:
        lines = f.readlines()
        # Loop through each line, get first and last digit, concatenate them, convert to int and add to Total
        for line in lines:
            digits = ''.join(x for x in line if x.isdigit())
            digit = str(digits[0]) + str(digits[len(digits)-1])
            total += int(digit)
            print(digits + ':' + digit)
                
    print(total)

if __name__ == '__main__':
    main()
