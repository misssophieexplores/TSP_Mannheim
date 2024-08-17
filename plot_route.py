import folium
from geopy.geocoders import Nominatim
from shapely.geometry import LineString

# Step 1: Initialize Nominatim API
geolocator = Nominatim(user_agent="mannheim-squares")

# Step 2: Define square names
square_names = ['A1 Mannheim', 'L1 Mannheim', 'B6 Mannheim']

# Step 3: Fetch coordinates for each square
square_coordinates = {}
for square in square_names:
    location = geolocator.geocode(square)
    if location:
        square_coordinates[square.split()[0]] = (location.latitude, location.longitude)

# Step 4: Define the route using the square names
route_squares = ['A1', 'L1', 'B6', 'A1']  # Example route

# Step 5: Extract the points corresponding to the route
route_points = [square_coordinates[square] for square in route_squares]

# Step 6: Create a map centered around Mannheim
# Center the map on the first square in the route
m = folium.Map(location=route_points[0], zoom_start=15)

# Step 7: Add squares to the map
for square, coords in square_coordinates.items():
    folium.Marker(
        location=coords,
        popup=square,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Step 8: Add the route to the map
folium.PolyLine(route_points, color="red", weight=2.5, opacity=1).add_to(m)

# Step 9: Save the map to an HTML file and display it
m.save('mannheim_route.html')

# Display the map in the notebook (if using Jupyter)
m
