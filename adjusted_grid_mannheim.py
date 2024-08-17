def location_to_coordinates(location_with_corner):
    # Split the location into the square name and the corner identifier
    location, corner = location_with_corner.split()
    
    # Maps for converting letters and numbers to grid coordinates
    letter_to_index_left = {letter: index for index, letter in enumerate("ABCDEFGHIK")}
    letter_to_index_right = {letter: index for index, letter in enumerate("LMNOPQRSTU")}
    
    # Extract the letter and number from the location
    letter, number = location[0], int(location[1])
    
    # Determine if the location is on the left or right of the castle
    if letter in letter_to_index_left:
        x_base = letter_to_index_left[letter]
        y_base = 7 - number  # Reverse the order of numbers to fit the grid orientation
    elif letter in letter_to_index_right:
        x_base = letter_to_index_right[letter]  # The x-coordinates are correct, starting from 0 for L
        y_base = 6 + number 
    else:
        raise ValueError("Invalid location")
    
    # Define the coordinates for each corner
    corners = {
        "ld": (x_base, y_base),       # left-down (bottom-left)
        "lu": (x_base, y_base + 1),   # left-up (top-left)
        "rd": (x_base + 1, y_base),   # right-down (bottom-right)
        "ru": (x_base + 1, y_base + 1) # right-up (top-right)
    }
    
    # Return the coordinate for the specified corner
    if corner in corners:
        return corners[corner]
    else:
        raise ValueError("Invalid corner specified. Use 'ld', 'lu', 'rd', or 'ru'.")




location_1 = "A7 ld"
location_2 = "A7 rd"
location_3 = "A6 lu"
location_4 = "A6 ru"

print(location_1, location_to_coordinates(location_1))
print(location_2, location_to_coordinates(location_2))
print(location_3, location_to_coordinates(location_3))
print(location_4, location_to_coordinates(location_4))
