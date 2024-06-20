# Define a global variable
global_var = 10

def print_global():
    # Access the global variable
    print(f"The global variable is: {global_var}")

def modify_global():
    # Modify the global variable
    global global_var
    global_var = 20
    print(f"The global variable has been modified to: {global_var}")

# Call the functions
print_global()  # Output: The global variable is: 10
modify_global() # Output: The global variable has been modified to: 20
print_global()  # Output: The global variable is: 20
