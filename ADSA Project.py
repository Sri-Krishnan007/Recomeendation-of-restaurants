import streamlit as st
import pandas as pd
from geopy.distance import geodesic
import numpy as np
import requests
import streamlit as st
from math import sin, cos, sqrt, atan2, radians
from PIL import Image
import webbrowser
import geocoder  
import networkx as nx

# --- Data Loading and Preprocessing ---
# Read data from Excel file
file_path = r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\ADSA\ADSA PR\Employee.xlsx"
df_employee = pd.read_excel(file_path)

# Load the CSV data for Chennai and Coimbatore
file_path_chennai = r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\ADSA\ADSA PR\Zomato Chennai Listing 2020.csv"
file_path_coimbatore = r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\ADSA\ADSA PR\Coimbatore Restaraunts.csv"
df_chennai = pd.read_csv(file_path_chennai)
df_coimbatore = pd.read_csv(file_path_coimbatore)

# --- Helper Functions ---
# Calculate the haversine distance between two points on Earth
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Define a graph structure
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = np.zeros((vertices, vertices))

    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight
        self.graph[v][u] = weight

# Prim's algorithm to find minimum spanning tree (MST)
def prim_mst(graph):
    parent = [-1] * graph.V
    key = [float('inf')] * graph.V
    mst_set = [False] * graph.V

    key[0] = 0  # Starting node
    parent[0] = -1

    for _ in range(graph.V):
        u = min_key(key, mst_set)
        mst_set[u] = True
        for v in range(graph.V):
            if graph.graph[u][v] > 0 and not mst_set[v] and key[v] > graph.graph[u][v]:
                key[v] = graph.graph[u][v]
                parent[v] = u

    return parent

def min_key(key, mst_set):
    min_val = float('inf')
    min_index = -1
    for v in range(len(key)):
        if key[v] < min_val and not mst_set[v]:
            min_val = key[v]
            min_index = v
    return min_index

# Convert dataframe to graph
def create_graph_from_df(data):
    num_vertices = len(data)
    graph = Graph(num_vertices)
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            distance = haversine_distance(data.iloc[i]['lat'], data.iloc[i]['lng'], data.iloc[j]['lat'], data.iloc[j]['lng'])
            graph.add_edge(i, j, distance)
    return graph

# Get user location using geocoder
def get_user_location():
    g = geocoder.ip('me')
    location = g.latlng
    if location:
        return location[0], location[1]
    return None

# Simple filtering and sorting to recommend restaurants within price limit and high combined rating
def chennai_restaurants_page():
    st.title("Chennai Restaurants")

    # Display the image at the top of the page
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Chennai Restaurants")

    # Sidebar with options
    st.sidebar.header('Filter Options')

    # Filter by location
    locations = df_chennai['Location'].unique()
    selected_location = st.sidebar.selectbox('Select Location', locations)

    # Filter by cuisine
    cuisines = df_chennai['Cuisine'].unique()
    selected_cuisine = st.sidebar.selectbox('Select Cuisine', cuisines)

    # Filter by price
    max_price = st.sidebar.number_input('Maximum Price for 2', value=1000)

    # Apply filters
    filtered_df = df_chennai[(df_chennai['Location'] == selected_location) &
                             (df_chennai['Cuisine'] == selected_cuisine) &
                             (df_chennai['Price for 2'] <= max_price)]

    # Drop 'S.No', 'Zomato URL', and 'Location' columns if they exist
    columns_to_drop = ['S.No', 'Zomato URL', 'Location']
    filtered_df = filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns])

    # Create combined rating
    filtered_df['combined_rating'] = filtered_df[['Dining Rating', 'Delivery Rating']].mean(axis=1)

    # Sort the restaurants by combined rating
    sorted_restaurants = filtered_df.sort_values(by='combined_rating', ascending=False)

    # Display filtered results
    st.write('## Filtered Results')
    st.write(sorted_restaurants)

# Recommend restaurants based on user's location
def restaurant_recommendation(user_lat, user_lon, data):
    try:
        # Create graph from dataframe
        graph = create_graph_from_df(data)

        # Run Prim's algorithm to find MST
        parent = prim_mst(graph)

        # Find restaurants connected to starting node (user location)
        connected_restaurants = []
        for i in range(1, len(parent)):
            if parent[i] == 0:
                connected_restaurants.append((data.iloc[i]['name'], haversine_distance(user_lat, user_lon, data.iloc[i]['lat'], data.iloc[i]['lng'])))

        # Sort restaurants by distance
        connected_restaurants.sort(key=lambda x: x[1])
        return connected_restaurants

    except FileNotFoundError:
        st.error("Error: CSV file not found.")
    except ValueError:
        st.error("Error: Invalid data in CSV file.")

# Display the recommendation page
def recommend_nearby_restaurants_page():
    st.title("Recommend Nearby Restaurants")
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Restaurant Recommendation")

    user_location = get_user_location()
    if user_location:
        user_lat, user_lon = user_location
        st.success(f"Your location is: Latitude: {user_lat}, Longitude: {user_lon}")
        recommended_restaurants = restaurant_recommendation(user_lat, user_lon, df_coimbatore)
        if recommended_restaurants:
            st.write("Recommended restaurants near your location:")
            for restaurant, distance in recommended_restaurants:
                st.write(f"- {restaurant} ({distance:.2f} km away)")
        else:
            st.warning("No restaurants found near your location.")
    else:
        st.warning("Unable to detect your location.")

# Sort restaurant data by star count
def sort_data_by_rating(data):
    sorted_data = data.sort_values(by='star_count', ascending=False, ignore_index=True)
    return sorted_data.to_dict('records')

# Display top restaurants sorted by star count
def review_rating_page():
    st.title("Review and Rating")
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Review and Rating")

    data = df_coimbatore
    sorted_data = sort_data_by_rating(data)
    for restaurant in sorted_data:
        name = restaurant['name']
        star_count = restaurant['star_count']
        st.write(f"Name: {name}, Star Rating Out of 5: {star_count}")

# Retrieve restaurant details from DataFrame
def get_restaurant_details(data, restaurant_name):
    selected_restaurant = data[data['name'] == restaurant_name]
    return (selected_restaurant.to_dict('records')[0] if not selected_restaurant.empty else None)

# Display selected restaurant details inside a box
def select_restaurant_page():
    st.title("Select Restaurant")
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Select Restaurant")

    data = df_coimbatore

    # Display a list of all restaurants
    restaurants = data['name'].tolist()
    selected_restaurant = st.selectbox("Choose a restaurant", restaurants)

    if selected_restaurant:
        details = get_restaurant_details(data, selected_restaurant)
        if details:
             st.markdown(
            f"""
            <div>
                <h3>{details['name']}</h3>
                <p><strong>Address:</strong> {details['address']}</p>
                <p><strong>Phone:</strong> {details['phone']}</p>
                <p><strong>Primary Category:</strong> {details['primary_category_name']}</p>
                <p><strong>Category:</strong> {details['category_name']}</p>
                <p><strong>Cuisine:</strong> {details['Cuisine']}</p>
                <p><strong>Top Dishes:</strong> {''.join(details['Top Dishes'])}</p>
                <p><strong>Price for 2:</strong> {details['Price for 2']}</p>
                <p><strong>Dining Rating:</strong> {details['Dining Rating']} / 5</p>
                <p><strong>Dining Rating Count:</strong> {details['Dining Rating Count']}</p>
                <p><strong>Delivery Rating:</strong> {details['Delivery Rating']} / 5</p>
                <p><strong>Delivery Rating Count:</strong> {details['Delivery Rating Count']}</p>
                <p><strong>Features:</strong> {''.join(details['Features'])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Find user's location and display it
def find_location_page():
    st.title("Find Your Location")
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Find Your Location")

    user_location = get_user_location()
    if user_location:
        user_lat, user_lon = user_location
        st.success(f"Your location is: Latitude: {user_lat}, Longitude: {user_lon}")
    else:
        st.warning("Unable to detect your location.")

# Open Google Maps
def open_maps_page():
    st.title("Open Maps")
    image = Image.open(r"C:\Users\srikr\Desktop\COLLEGE\Sem 4\Predictive analysis\PA Project\fac.png")
    st.image(image, caption="Open Maps")

    map_url = ("https://www.google.com/maps/d/u/0/"
               "embed?mid=1nqmVv1_AgLWHO10knGd26DFtuUQ1Gfk&ehbc=2E312F")
    if st.button("Open Google Maps"):
        webbrowser.open(map_url)

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Restaurant Recommendation System", page_icon=":fork_and_knife:")
    pages = {
        "Find Your Location": find_location_page,
        "Open Maps": open_maps_page,
        "Recommend Nearby Restaurants": recommend_nearby_restaurants_page,
        "Review and Rating": review_rating_page,
        "Select Restaurant": select_restaurant_page,
        "Chennai Restaurants": chennai_restaurants_page,
        "Employee Route Optimization": employee_route_optimization_page
    }
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

# --- Employee Route Optimization ---
def employee_route_optimization_page():
    st.title("Employee Route Optimization")

    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file with employee data", type=["xlsx"])

    if uploaded_file is not None:
        df_employee = pd.read_excel(uploaded_file)

        # Ensure the Excel file has the required columns
        if 'Locations' in df_employee.columns and 'Latitude' in df_employee.columns and 'Longitude' in df_employee.columns:
            # Extract the location names, latitude, and longitude values
            locations = df_employee[['Locations', 'Latitude', 'Longitude']].values
            num_locations = len(locations)

            # Initialize the distance matrix with infinities
            distance_matrix = np.full((num_locations, num_locations), np.inf)

            # Fill the distance matrix with geodesic distances
            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j:
                        loc1 = (locations[i][1], locations[i][2])
                        loc2 = (locations[j][1], locations[j][2])
                        distance_matrix[i][j] = geodesic(loc1, loc2).kilometers

            # Create a graph using NetworkX
            G = nx.Graph()

            # Add nodes with positions
            for i, loc in enumerate(locations):
                G.add_node(i, pos=(loc[2], loc[1]), label=loc[0])

            # Add edges with weights (distances)
            for i in range(num_locations):
                for j in range(num_locations):
                    if i != j and distance_matrix[i][j] < np.inf:
                        G.add_edge(i, j, weight=distance_matrix[i][j])

            # Calculate the shortest path for each pair of locations using Dijkstra's algorithm
            shortest_paths = {}
            for i in range(num_locations):
                for j in range(i + 1, num_locations):
                    shortest_path = nx.shortest_path(G, source=i, target=j, weight='weight')
                    shortest_distance = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                    shortest_paths[(i, j)] = {'path': shortest_path, 'distance': shortest_distance}

            # Sort the paths based on their total distance
            sorted_paths = sorted(shortest_paths.items(), key=lambda x: x[1]['distance'])

            # Display the top 5 best routes
            st.subheader("Top 5 Best Routes:")
            for i, (path, info) in enumerate(sorted_paths[:5], start=1):
                start_location = locations[path[0]][0]
                end_location = locations[path[1]][0]
                st.write(f"Route {i}: {start_location} -> {end_location}, Distance: {info['distance']} km, Path: {info['path']}")

        else:
            st.error("The Excel file does not contain 'Locations', 'Latitude', and 'Longitude' columns.")
    else:
        st.info("Please upload an Excel file with employee data.")

if __name__ == "__main__":
    main()
