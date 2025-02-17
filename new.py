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
import mysql.connector
import threading
from threading import Thread

background_task_started = False

DB_CONFIG = {
    "host": "localhost",
    "user": "root",      
    "password": "2004",  
    "database": "traffic_data"
}


def init_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                origin_latitude DOUBLE,
                origin_longitude DOUBLE,
                green_duration DOUBLE,
                red_duration DOUBLE
            )
        """)
        conn.commit()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")

app = Flask(__name__,static_folder='static')
CORS(app)

GOOGLE_MAPS_API_KEY = 'Your_Maps_API_Key'


green_light_model_path = os.path.join('models','xgb_model_green.pkl')
red_light_model_path = os.path.join('models','xgb_model_red.pkl')

try:
    green_light_model = joblib.load(green_light_model_path)
    red_light_model = joblib.load(red_light_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    green_light_model = None
    red_light_model = None

def validate_login(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM login_data WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        if user:
            if user['password'] == password:
                return True, None 
            else:
                return False, "password"  
        else:
            return False, "username"  
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return False, "database_error"

def save_to_database(origin_latitude, origin_longitude, green_duration, red_duration):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        timestamp = datetime.now()
        cursor.execute("""
            INSERT INTO traffic_data (timestamp, origin_latitude, origin_longitude, green_duration, red_duration)
            VALUES (%s, %s, %s, %s, %s)
        """, (timestamp, origin_latitude, origin_longitude, green_duration, red_duration))
        conn.commit()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")

def save_to_csv_periodically(csv_path, interval=600):
    while True:
        time.sleep(interval)
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            query = "SELECT * FROM traffic_data"
            df = pd.read_sql(query, conn)
            conn.close()
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving to CSV: {e}")

def find_nearby_places(origin_latitude, origin_longitude, radius=1000):
    
    nearby_search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
  
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    
   
    nearby_place_route = {}
    
   
    nearby_params = {
        'location': f"{origin_latitude},{origin_longitude}",
        'radius': radius,
        'key': GOOGLE_MAPS_API_KEY
    }
    
    nearby_response = requests.get(nearby_search_url, params=nearby_params)
    nearby_data = nearby_response.json()
    
    closest_distance = float('inf')
    
    for place in nearby_data.get('results', []):
        place_lat = place['geometry']['location']['lat']
        place_lng = place['geometry']['location']['lng']
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  
            
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            return distance
        
        distance = haversine_distance(origin_latitude, origin_longitude, place_lat, place_lng)
 
        if distance <= radius and distance < closest_distance:
            
            route_params = {
                'origin': f"{origin_latitude},{origin_longitude}",
                'destination': f"{place_lat},{place_lng}",
                'key': GOOGLE_MAPS_API_KEY
            }
            
            route_response = requests.get(directions_url, params=route_params)
            route_data = route_response.json()
            
            if route_data.get('routes'):
                
                route = route_data['routes'][0]
                
                
                duration_value = float(route['legs'][0]['duration']['value']) / 60  
                
                closest_distance = distance
                nearby_place_route = {
                    'latitude': place_lat, 
                    'longitude': place_lng, 
                    'duration': duration_value, 
                    'distance': distance / 1000   }
    return nearby_place_route

def prepare_ml_features(origin_latitude, origin_longitude, destination_latitude, destination_longitude, duration, distance_km):
    
    now = datetime.now()
    
    # Prepare features
    features = [
        now.weekday(), 
        now.hour, 
        now.hour * 60 + now.minute,  
        duration * 60,
        distance_km,  
        origin_latitude,  
        origin_longitude,  
        destination_latitude,  
        destination_longitude  
    ]
    
    return np.array(features).reshape(1, -1)

def generate_ml_features(origin_latitude, origin_longitude, radius=1000):
    nearby_place = find_nearby_places(
        origin_latitude, 
        origin_longitude, 
        radius
    )
    
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

@app.before_request
def start_background_task():
    global background_task_started
    if not background_task_started:
        background_task_started = True
        Thread(
            target=save_to_csv_periodically,
            args=("traffic_data.csv",),
            daemon=True
        ).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    if request.method == 'POST':

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        is_valid, error = validate_login(username, password)
        if is_valid:
            return jsonify({
                'success': True,
                'message': 'Login successful'
            })
        else:
            return jsonify({
                'success': False,
                'error': error 
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
                'green_duration': 30,  
                'red_duration': 30   
            }), 500


        try:
           
            origin_latitude = float(request.form.get('latitude', 0))
            origin_longitude = float(request.form.get('longitude', 0))
            duration = int(request.form.get('duration', 10))
        
            features = generate_ml_features(origin_latitude, origin_longitude)
        
            predicted_green = green_light_model.predict(features[0])[0]
            predicted_red = red_light_model.predict(features[0])[0]

            save_to_database(
                origin_latitude, origin_longitude,
                round(max(10, predicted_green)), round(max(10, predicted_red))
            )

            return jsonify({
                'green_duration': round(max(10, predicted_green)),  
                'red_duration': round(max(10, predicted_red))       
            })
    
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({
                'error': str(e),
                'green_duration': 30,  
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
    init_database()
    app.run(debug='true')








