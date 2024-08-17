import numpy as np
import folium
from geopy.geocoders import Nominatim

# Cache to store precomputed coordinates for locations to avoid recalculating them multiple times
coordinates_cache = {}

def location_to_coordinates(location):
    """
    Converts a location in the grid (e.g., 'A1', 'B6', etc.) to latitude and longitude using Nominatim API.
    Uses a cache to store and retrieve precomputed coordinates for efficiency.
    
    Parameters:
    location (str): The location in the format 'LetterNumber' (e.g., 'A1').

    Returns:
    tuple: A tuple representing the (latitude, longitude) coordinates on the map.
    """

    # Check if the location's coordinates are already in the cache
    if location in coordinates_cache:
        return coordinates_cache[location]
    
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="mannheim-squares")

    # Append "Mannheim" to the location for accurate geocoding
    location_query = f"{location} Mannheim"

    # Fetch coordinates from Nominatim
    location_data = geolocator.geocode(location_query)
    if location_data:
        coordinates_cache[location] = (location_data.latitude, location_data.longitude)
    else:
        raise ValueError(f"Could not find location {location_query}")
    
    return coordinates_cache[location]

def calculate_manhattan_distance(location1, location2):
    """
    Calculates the Manhattan distance between two locations on the grid.
    Note: For geographical locations, this is just a placeholder.

    Parameters:
    location1 (str): The first location (e.g., 'A1').
    location2 (str): The second location (e.g., 'B6').

    Returns:
    float: The Manhattan distance between the two locations.
    """

    # Convert the locations to grid coordinates (lat/long)
    lat1, lon1 = location_to_coordinates(location1)
    lat2, lon2 = location_to_coordinates(location2)
    
    # Calculate the Manhattan distance (using lat/lon for simplicity)
    distance = abs(lat1 - lat2) + abs(lon1 - lon2)
    
    return distance

def create_distance_matrix(locations):
    """
    Creates a distance matrix for the given locations, using Manhattan distance.

    Parameters:
    locations (list): A list of location strings (e.g., ['A1', 'B6']).

    Returns:
    np.ndarray: A symmetric matrix where element (i, j) is the distance between locations[i] and locations[j].
    """

    k = len(locations)  # Number of locations
    distance_matrix = np.zeros((k, k))  # Initialize a kxk distance matrix
    
    for i in range(k):
        for j in range(i, k):  # Only calculate the upper triangle of the matrix to avoid redundancy
            if i != j:
                distance = calculate_manhattan_distance(locations[i], locations[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Use symmetry: distance(i, j) = distance(j, i)
            else:
                distance_matrix[i, j] = 0  # The distance to the same location is zero
    
    return distance_matrix

def simulated_annealing_minimization(distances, iterations):
    """
    Performs simulated annealing to find the route that minimizes the total distance.
    
    Parameters:
    distances (np.ndarray): A precomputed distance matrix.
    iterations (int): The number of iterations for the simulated annealing process.

    Returns:
    tuple: A tuple containing the history of distances and the history of routes over the iterations.
    """

    k = len(distances)  # Number of locations
    
    # Generate an initial random permutation of cities (initial route)
    x0 = np.random.permutation(k)
    
    def switch(x, i, j):
        """
        Swaps two elements in a route (used to generate neighboring solutions).
        
        Parameters:
        x (np.ndarray): The current route.
        i (int): Index of the first element to swap.
        j (int): Index of the second element to swap.

        Returns:
        np.ndarray: The new route after swapping.
        """
        x[i], x[j] = x[j], x[i]
        return x

    def calculate_total_distance(route):
        """
        Calculates the total distance for a given route, including returning to the starting point.
        
        Parameters:
        route (np.ndarray): An array representing the order of locations to visit.

        Returns:
        float: The total distance of the route.
        """
        total_distance = 0
        for i in range(1, k):
            total_distance += distances[route[i-1], route[i]]
        # Add distance from the last city back to the first city to complete the cycle
        total_distance += distances[route[-1], route[0]]
        return total_distance
    
    cstop = iterations  # Maximum number of iterations
    xn = x0.copy()  # Current route
    vector_distance = []  # History of distances
    mat_route = np.zeros((cstop, k), dtype=int)  # History of routes
    
    n = 0
    while n < cstop:
        n += 1
        
        # Calculate the total distance with the current route
        current_distance = calculate_total_distance(xn)
        
        while True:
            # Generate two random indices for swapping cities
            ij = np.random.choice(k, size=2, replace=False)
            if np.random.uniform() < 1 / (k * (k - 1) / 2):
                break
        
        # Find the indices of the selected cities in the route
        I = np.where(xn == ij[0])[0][0]
        J = np.where(xn == ij[1])[0][0]  # Corrected indexing here
        
        # Generate a neighboring route by switching two cities
        yn = switch(xn.copy(), I, J)
        
        # Calculate the total distance with the new route
        new_distance = calculate_total_distance(yn)
        
        # Acceptance probability (Minimization: prefer smaller distances)
        if new_distance < current_distance:
            a = 1  # Always accept if the new distance is smaller
        else:
            a = np.exp(-(new_distance - current_distance) / (1 + n))  # Probability of accepting a worse route
        
        # Decide whether to accept the new route
        if np.random.uniform() < a:
            xn = yn
        
        # Store the final distance and route for this iteration
        final_distance = calculate_total_distance(xn)
        vector_distance.append(final_distance)
        mat_route[n-1, :] = xn
    
    return vector_distance, mat_route

# Example locations
locations = ["A1", "B6", "L1", "L2", "R1", "R3"]

# Create the distance matrix for these locations using Manhattan distances
distance_matrix = create_distance_matrix(locations)
print("Distance Matrix:\n", distance_matrix)

# Run the simulated annealing algorithm to find the optimal route
iterations = 1000  # Number of iterations for the simulated annealing process
distances, routes = simulated_annealing_minimization(distance_matrix, iterations)

# Output the final distance and route
final_route = routes[-1]
final_distance = distances[-1]

# Map the route indices back to the location names
final_route_locations = [locations[i] for i in final_route]
# Add the starting location to the end of the route to complete the cycle
final_route_locations.append(final_route_locations[0])

print("Final distance:", final_distance)
print("Final route:", " -> ".join(final_route_locations))

# Plotting the final route on OpenStreetMap using folium
# Create a map centered around the first location
map_center = location_to_coordinates(final_route_locations[0])
m = folium.Map(location=map_center, zoom_start=15)

# Add markers for each location in the route
for loc in final_route_locations:
    coords = location_to_coordinates(loc)
    folium.Marker(location=coords, popup=loc, icon=folium.Icon(color='blue')).add_to(m)

# Add the route as a PolyLine
route_coords = [location_to_coordinates(loc) for loc in final_route_locations]
folium.PolyLine(route_coords, color="red", weight=2.5, opacity=1).add_to(m)

# Save the map to an HTML file and display it
m.save('mannheim_final_route.html')

