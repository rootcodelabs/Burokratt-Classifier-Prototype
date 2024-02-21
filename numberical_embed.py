class StringConverter:
    def __init__(self):
        pass

    def string_to_integer(self, input_string):
        """
        Convert a string to its numerical representation
        """
        numerical_representation = 0
        for char in input_string:
            numerical_representation = numerical_representation * 256 + ord(char)
        return numerical_representation

    def integer_to_string(self, input_integer):
        """
        Convert an integer to its string representation
        """
        string_representation = ""
        while input_integer > 0:
            string_representation = chr(input_integer % 256) + string_representation
            input_integer //= 256
        return string_representation

# Example usage:
converter = StringConverter()

# Convert string to integer
input_string = "Knight rider"
numerical_representation = converter.string_to_integer(input_string)
print(f"Numerical representation of '{input_string}': {numerical_representation}")

# Convert integer to string
input_integer = numerical_representation
string_representation = converter.integer_to_string(input_integer)
print(f"String representation of {input_integer}: {string_representation}")
