import random
import string

def generate_password(min_length, numbers=True, special_char=True):
    letters = string.ascii_letters
    digits = string.digits
    special = string.punctuation

    characters = letters
    if numbers:
        characters += digits
    if special_char:
        characters += special
    
    password = ""
    meets_criteria = False
    has_number = False
    has_special = False

    while not meets_criteria or len(password) < min_length:
        new_char = random.choice(characters)
        password += new_char

        if new_char in digits:
            has_number = True
        if new_char in special:
            has_special = True
        
        meets_criteria = True
        if numbers:
            meets_criteria = has_number
        if special_char:
            meets_criteria = meets_criteria and has_special
    
    return password

min_length = int(input("Please enter minimum length of the password: "))
needs_number = input("Does the password need numbers (y/n)? ").lower() == "y"
needs_special = input("Does the password need numbers (y/n)? ").lower() == "y"
pwd = generate_password(min_length, needs_number, needs_special)
print(f"Your generated password is: {pwd}")