import requests
import googlemaps
import csv
from datetime import datetime

def get_distance_matrix(api_key, origin, destination, traffic_model):
    endpoint = 'https://maps.googleapis.com/maps/api/distancematrix/json'
    params = {
        'origins': origin,
        'destinations': destination,
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

def calculate_traffic_density(duration_in_traffic_sec, distance_km):
    """Calculate traffic density (vehicles per km)."""
    # Higher duration or slower speed indicates higher density
    # We'll use a simple approximation where high traffic duration means high density
    if distance_km > 0:
        density = duration_in_traffic_sec / (distance_km * 60)  # Vehicles per minute per km
        return round(density, 2)
    return 0

def calculate_light_durations(duration_in_traffic_sec, base_green=30, base_red=30):
    """Calculate green and red light durations based on traffic conditions."""
    green_time = base_green + (duration_in_traffic_sec / 60)  # 1 min delay -> +1 sec green
    red_time = base_red + max(0, 10 - (duration_in_traffic_sec / 60))  # Adjust based on delay
    return round(green_time, 2), round(red_time, 2)

def extract_info(data, traffic_model, api_key):
    if not data or 'rows' not in data:
        return None
    
    element = data['rows'][0]['elements'][0]
    origin_address = data['origin_addresses'][0]
    destination_address = data['destination_addresses'][0]
    
    # Get latitude and longitude for origin and destination
    origin_lat, origin_lng = get_lat_lng(api_key, origin_address)
    destination_lat, destination_lng = get_lat_lng(api_key, destination_address)
    
    # Convert duration_in_traffic to seconds
    duration_in_traffic_text = element.get('duration_in_traffic', {}).get('value', 0)  # in seconds
    distance_km = element['distance']['value'] / 1000  # Convert meters to kilometers
    
    # Calculate traffic density
    traffic_density = calculate_traffic_density(duration_in_traffic_text, distance_km)
    
    # Calculate green and red light durations
    green_time, red_time = calculate_light_durations(duration_in_traffic_text)
    
    return {
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
        'green_light': green_time,
        'red_light': red_time
    }

def process_csv(input_csv, output_csv, api_key):
    # Open the input CSV file
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    # Prepare output data
    output_data = []
    
    # Traffic models
    traffic_models = ['best_guess', 'optimistic', 'pessimistic']
    
    for row in rows:
        origin = row['origin']
        destination = row['destination']
        
        for model in traffic_models:
            data = get_distance_matrix(api_key, origin, destination, model)
            if data:
                result = extract_info(data, model, api_key)
                if result:
                    # Add the current date and time to the result
                    now = datetime.now()
                    result['current_time'] = now.strftime("%H:%M:%S")
                    result['current_date'] = now.strftime("%d-%m-%Y")
                    result['input_origin'] = origin
                    result['input_destination'] = destination
                    output_data.append(result)
    
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
    input_csv = 'input.csv'  # Contains 'origin' and 'destination' columns
    output_csv = 'output_traffic_data.csv'
    
    # Process the CSV files
    process_csv(input_csv, output_csv, api_key)

if __name__ == "__main__":
    main()
