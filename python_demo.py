###########################################################
# This script demonstrates some basic python functionality
###########################################################

# This is a comment

# Print a message to the console
print('Hello world!')

# Declare a function to print input from the user
def print_input():
    # Get input from the user and store it in the 'name' string variable
    name = input('Please type your name and press return: ')
    print(f'Hello, {name}!')

# Call the function
print_input()

# Declare a numeric variable
x = 2
print(f"x = {x}")

y = input("Please provide a numeric value for y: ")
y = int(y) # convert from a string to a numeric

# Compute the sum of x and y
a = x + y
print(f"x + y = a = {a}")

# Compute the product
b = x * y
print(f"x * y = b = {b}")

# Manually declare a list containing the
# first 8 numbers of the fibonacci sequence
fm = [0, 1, 1, 2, 3, 5, 8, 13]
print(f"Fibonacci (A): {fm}")

n = int(input("Please provide a length for the fibonacci sequence: "))

# Declare a function to calculate the fibonacci sequence automatically (the sequence in which each number is the sum of the two preceding ones)
def fibonacci(count):
    sequence = [0, 1] # Initialize the sequence as a list with the first two values, 0 and 1.

    # Continue generating the sequence until it reaches the desired length (count).
    while len(sequence) < count:
        # Calculate the next value by adding the last two values in the sequence
        next_value = sequence[len(sequence) - 1] + sequence[len(sequence) - 2]
        # Append the next value to the sequence.
        sequence.append(next_value)
    return sequence # Return the sequence to the caller

# Call the function
fa = fibonacci(count=n)
print(f"Fibonacci (B): {fa}")
