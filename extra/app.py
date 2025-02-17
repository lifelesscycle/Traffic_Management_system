from flask import Flask, render_template, request, jsonify, url_for
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
from datetime import datetime
import uuid
import time
import math

app = Flask(__name__,static_folder='static')
CORS(app)

GOOGLE_MAPS_API_KEY = 'AIzaSyADBKX48xzDBlvFbImdpfjG4_hL-gZ2GpU'


green_light_model_path = os.path.join('models','xgb_model_green.pkl')
red_light_model_path = os.path.join('models','xgb_model_red.pkl')

# Load pre-trained ML models
try:
    green_light_model = joblib.load(green_light_model_path)
    red_light_model = joblib.load(red_light_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    green_light_model = None
    red_light_model = None

def find_nearby_places(origin_latitude, origin_longitude, radius=1000):
    # Google Places API endpoint for nearby search
    nearby_search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    # Google Directions API endpoint for route calculation
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Nearest place to return
    nearby_place_route = {}
    
    # Parameters for nearby search
    nearby_params = {
        'location': f"{origin_latitude},{origin_longitude}",
        'radius': radius,
        'key': GOOGLE_MAPS_API_KEY
    }
    
    # Make nearby search request
    nearby_response = requests.get(nearby_search_url, params=nearby_params)
    nearby_data = nearby_response.json()
    
    # Track the closest place
    closest_distance = float('inf')
    
    # Process each nearby place
    for place in nearby_data.get('results', []):
        # Extract place coordinates
        place_lat = place['geometry']['location']['lat']
        place_lng = place['geometry']['location']['lng']
        
        # Haversine distance calculation
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            
            # Convert latitude and longitude to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            return distance
        
        # Calculate distance
        distance = haversine_distance(origin_latitude, origin_longitude, place_lat, place_lng)
        
        # Only process places within the specified radius
        if distance <= radius and distance < closest_distance:
            # Get route details
            route_params = {
                'origin': f"{origin_latitude},{origin_longitude}",
                'destination': f"{place_lat},{place_lng}",
                'key': GOOGLE_MAPS_API_KEY
            }
            
            route_response = requests.get(directions_url, params=route_params)
            route_data = route_response.json()
            
            # Check if a route exists
            if route_data.get('routes'):
                # Extract route information
                route = route_data['routes'][0]
                
                # Parse duration to get numeric value
                duration_value = float(route['legs'][0]['duration']['value']) / 60  # convert to minutes
                
                # Update closest place
                closest_distance = distance
                nearby_place_route = {
                    'latitude': place_lat, 
                    'longitude': place_lng, 
                    'duration': duration_value, 
                    'distance': distance / 1000  # convert distance to km
                }
    
    return nearby_place_route

def prepare_ml_features(origin_latitude, origin_longitude, destination_latitude, destination_longitude, duration, distance_km):
    # Get current timestamp details
    now = datetime.now()
    
    # Prepare features
    features = [
        now.weekday(),  # day of week (0-6)
        now.hour,  # hour of day (0-23)
        now.hour * 60 + now.minute,  # minute of day (0-1439)
        duration * 60,  # duration in traffic (seconds)
        distance_km,  # distance in kilometers
        origin_latitude,  # origin latitude
        origin_longitude,  # origin longitude
        destination_latitude,  # destination latitude
        destination_longitude  # destination longitude
    ]
    
    return np.array(features).reshape(1, -1)

def generate_ml_features(origin_latitude, origin_longitude, radius=1000):
    # Find the nearest place with route information
    nearby_place = find_nearby_places(
        origin_latitude, 
        origin_longitude, 
        radius
    )
    
    # Prepare ML features for the nearest place
    ml_features = []
    if nearby_place:
        features = prepare_ml_features(
            origin_latitude, 
            origin_longitude, 
            nearby_place['latitude'], 
            nearby_place['longitude'], 
            nearby_place['duration'], 
            nearby_place['distance']
        )
        ml_features.append(features)
    
    return ml_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    if request.method == 'POST':
        # Get JSON data from the request
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Here you would typically:
        # 1. Check against a database of users
        # 2. Verify credentials
        # For this example, we'll use a simple hard-coded check
        if username == 'admin' and password == 'password':
            return jsonify({
                'success': True,
                'message': 'Login successful'
            })
        else:
            # Determine specific error
            if username != 'admin':
                return jsonify({
                    'success': False,
                    'error': 'username'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'password'
                })
            
@app.route('/dashboard',methods=['GET','POST'])
def dashboard():
    time.sleep(1)
    return render_template('dashboard.html')

@app.route('/traffic_prediction', methods=['GET','POST'])
def traffic_prediction():
    if request.method == 'POST':
        if green_light_model is None or red_light_model is None:
            return jsonify({
                'error': 'ML models not loaded properly',
                'green_duration': 30,  # Default fallback
                'red_duration': 30     # Default fallback
            }), 500


        try:
            # Extract parameters from form data
            origin_latitude = float(request.form.get('latitude', 0))
            origin_longitude = float(request.form.get('longitude', 0))
            duration = int(request.form.get('duration', 10))
        
        # Prepare features
            features = generate_ml_features(origin_latitude, origin_longitude)
        
        # Predict green and red light durations
            predicted_green = green_light_model.predict(features[0])[0]
            predicted_red = red_light_model.predict(features[0])[0]
        
            return jsonify({
                'green_duration': round(max(10, predicted_green)),  # Ensure minimum duration
                'red_duration': round(max(10, predicted_red))       # Ensure minimum duration
            })
    
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({
                'error': str(e),
                'green_duration': 30,  # Fallback values
                'red_duration': 30
            }), 400
    else :
        time.sleep(1)
        return render_template('traffic_prediction.html')
    
traffic_emergency={}

@app.route('/emergency',methods=['GET','POST'])
def emergency():
    if request.method == 'POST':
       
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data received'
            }), 400
        
        traffic_emergency_data = {
            'emergency' : data.get('emergency',''),
            'location' : data.get('location',''),
            'city' : data.get('city',''),
            'state' : data.get('state',''),
            'message' : data.get('message','')
        }

        emergency_id = str(uuid.uuid4())
        traffic_emergency[emergency_id]=traffic_emergency_data
        
        return jsonify({
            'status': 'success',
            'message': 'Traffic emergency submitted successfully',
            'emergency_id': emergency_id
    }),201
    time.sleep(1)
    return render_template('emergency.html')
    
@app.route('/display_emergency_list',methods=['GET','POST'])
def display_emergency_list():
    time.sleep(1)
    return render_template('display_emergency_list.html', emergencies=traffic_emergency)

@app.route('/add_location')
def add_location():
    time.sleep(1.5)
    return render_template('add_location.html')

@app.route('/route_optimizer',methods=['GET','POST'])
def route_optimizer():
    if request.method == 'POST':
        data = request.json
        start_location=data.get('start')
        end_location = data.get('end')
        start_coords = geocode_location(start_location)
        end_coords = geocode_location(end_location)
    
        if not start_coords or not end_coords:
            return jsonify({'error': 'Could not geocode locations'}), 400
    
        return jsonify({
            'start_coords': start_coords,
            'end_coords': end_coords
        })
    else :
        time.sleep(1)
        return render_template('route_optimizer.html')
  
def geocode_location(location):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': location,
        'key': GOOGLE_MAPS_API_KEY
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return f"{location['lat']},{location['lng']}"
    else:
        return None


if __name__ == '__main__' :
    app.run(debug='true')

