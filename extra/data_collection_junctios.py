import requests
import csv
from datetime import datetime

def get_distance_matrix(api_key, origins, destinations, traffic_model):
    """Get traffic duration between origins and destinations using Google Distance Matrix API."""
    endpoint = 'https://maps.googleapis.com/maps/api/distancematrix/json'
    params = {
        'origins': '|'.join(origins),
        'destinations': '|'.join(destinations),
        'departure_time': 'now',
        'traffic_model': traffic_model,
        'key': api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error ({traffic_model}): {response.status_code}")
        return None

def calculate_traffic_density(duration_in_traffic_sec, distance_km):
    """Calculate traffic density (vehicles per km)."""
    if distance_km > 0:
        density = duration_in_traffic_sec / (distance_km * 60)  # Vehicles per minute per km
        return round(density, 2)
    return 0

def calculate_signal_durations(duration_in_traffic_sec, base_green=30, base_red=30):
    """Calculate green and red signal durations based on traffic duration."""
    green_time = base_green + (duration_in_traffic_sec / 60)  # 1 min delay -> +1 sec green
    red_time = base_red + max(0, 10 - (duration_in_traffic_sec / 60))  # Adjust based on delay
    return round(green_time, 2), round(red_time, 2)

def get_lat_lng(api_key, address):
    """Get latitude and longitude for a given address using the Geocoding API."""
    endpoint = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {
        'address': address,
        'key': api_key
    }
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
    print(f"Error retrieving geolocation for {address}: {response.status_code}")
    return None, None

def extract_info(data, traffic_model, directions, api_key):
    """Extract necessary data from the response and calculate required info."""
    if not data or 'rows' not in data:
        return None
    
    elements = data['rows'][0]['elements']
    output_data = []
    
    for i, direction in enumerate(directions):
        element = elements[i]
        origin_address = direction
        destination_address = data['destination_addresses'][0]
        
        # Get latitude and longitude for origin and destination
        origin_lat, origin_lng = get_lat_lng(api_key, origin_address)
        destination_lat, destination_lng = get_lat_lng(api_key, destination_address)
        
        # Convert duration_in_traffic to seconds
        duration_in_traffic_text = element.get('duration_in_traffic', {}).get('value', 0)  # in seconds
        distance_km = element['distance']['value'] / 1000  # Convert meters to kilometers
        
        # Calculate traffic density
        traffic_density = calculate_traffic_density(duration_in_traffic_text, distance_km)
        
        # Calculate green and red signal durations
        green_time, red_time = calculate_signal_durations(duration_in_traffic_text)
        
        output_data.append({
            'traffic_model': traffic_model,
            'origin_address': origin_address,
            'destination_address': destination_address,
            'origin_lat': origin_lat,
            'origin_lng': origin_lng,
            'destination_lat': destination_lat,
            'destination_lng': destination_lng,
            'distance': element['distance']['text'],
            'duration': element['duration']['text'],
            'duration_in_traffic': element.get('duration_in_traffic', {}).get('text', 'N/A'),
            'traffic_density': traffic_density,
            'green_signal': green_time,
            'red_signal': red_time,
            'direction': direction
        })
    
    return output_data

def process_junction_traffic(input_csv, output_csv, api_key):
    """Process CSV of junction data and calculate traffic data."""
    # Open the input CSV file
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    # Prepare output data
    output_data = []
    
    # Traffic models
    traffic_models = ['best_guess', 'optimistic', 'pessimistic']
    
    # Define the 4 directions around the junction (N, S, E, W)
    directions = ['North', 'South', 'East', 'West']
    
    for row in rows:
        origin_addresses = [row['origin_N'], row['origin_S'], row['origin_E'], row['origin_W']]
        destination_address = row['destination']  # The common destination (junction)
        
        for model in traffic_models:
            data = get_distance_matrix(api_key, origin_addresses, [destination_address]*4, model)
            if data:
                result = extract_info(data, model, directions, api_key)
                if result:
                    # Add the current date and time to the result
                    now = datetime.now()
                    for res in result:
                        res['current_time'] = now.strftime("%H:%M:%S")
                        res['current_date'] = now.strftime("%d-%m-%Y")
                        output_data.append(res)
    
    # Write to the output CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=output_data[0].keys())
        writer.writeheader()
        writer.writerows(output_data)
    print(f"Results saved to {output_csv}")

def main():
    # Your Google Maps API key
    api_key = 'AIzaSyADBKX48xzDBlvFbImdpfjG4_hL-gZ2GpU'
    
    # Input and output CSV files
    input_csv = 'junction_input.csv'  # Contains 'origin_N', 'origin_S', 'origin_E', 'origin_W', 'destination'
    output_csv = 'junction_traffic_output.csv'
    
    # Process the junction traffic data
    process_junction_traffic(input_csv, output_csv, api_key)

if __name__ == "__main__":
    main()
