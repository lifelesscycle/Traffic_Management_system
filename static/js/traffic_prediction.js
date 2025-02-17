document.addEventListener('DOMContentLoaded', () => {
    let map;
    let marker;
    let geocoder;
    let placesService;

    // Initialize Google Maps
    function initMap() {
        // Default location (e.g., New York City)
        const defaultLocation = { lat: 40.7128, lng: -74.0060 };

        map = new google.maps.Map(document.getElementById('map'), {
            center: defaultLocation,
            zoom: 10,
            mapTypeControl: false,
            streetViewControl: false
        });

        geocoder = new google.maps.Geocoder();
        placesService = new google.maps.places.PlacesService(map);
    }

    // Find and display location
    function findLocation() {
        const locationInput = document.getElementById('locationInput');
        const durationInput = document.getElementById('durationInput');
        const coordinatesDisplay = document.getElementById('coordinates');
        const placeDetailsDisplay = document.getElementById('placeDetails');
        const trafficResultsDisplay = document.getElementById('trafficResults');

        const location = locationInput.value.trim();
        const duration = durationInput.value.trim();

        // Reset previous displays
        coordinatesDisplay.innerHTML = 'Loading location details...';
        placeDetailsDisplay.innerHTML = 'Fetching place information...';
        trafficResultsDisplay.innerHTML = 'Predicting traffic durations...';

        if (!location || !duration) {
            coordinatesDisplay.textContent = 'Please enter a location and duration';
            return;
        }

        // Clear previous marker
        if (marker) {
            marker.setMap(null);
        }

        // Geocode the location
        geocoder.geocode({ address: location }, (results, status) => {
            if (status === 'OK') {
                const place = results[0];
                
                // Center map and add marker
                map.setCenter(place.geometry.location);
                map.setZoom(15);

                marker = new google.maps.Marker({
                    map: map,
                    position: place.geometry.location
                });

                // Display coordinates and basic info
                coordinatesDisplay.innerHTML = `
                    <strong>Location:</strong> ${place.formatted_address}<br>
                    <strong>Latitude:</strong> ${place.geometry.location.lat()}<br>
                    <strong>Longitude:</strong> ${place.geometry.location.lng()}
                `;

                // Fetch additional place details
                placesService.getDetails(
                    {
                        placeId: place.place_id,
                        fields: ['name', 'formatted_address', 'international_phone_number', 'website', 'types', 'rating', 'user_ratings_total']
                    },
                    (placeDetails, detailStatus) => {
                        if (detailStatus === google.maps.places.PlacesServiceStatus.OK) {
                            let detailsHTML = `
                                <p><strong>Name:</strong> ${placeDetails.name || 'N/A'}</p>
                                <p><strong>Address:</strong> ${placeDetails.formatted_address || 'N/A'}</p>
                                <p><strong>Phone:</strong> ${placeDetails.international_phone_number || 'N/A'}</p>
                                <p><strong>Website:</strong> ${placeDetails.website ? `<a href="${placeDetails.website}" target="_blank">Visit Website</a>` : 'N/A'}</p>
                                <p><strong>Types:</strong> ${placeDetails.types ? placeDetails.types.join(', ') : 'N/A'}</p>
                                <p><strong>Rating:</strong> ${placeDetails.rating ? `${placeDetails.rating}/5 (${placeDetails.user_ratings_total} reviews)` : 'N/A'}</p>
                            `;
                            placeDetailsDisplay.innerHTML = detailsHTML;
                        }
                    }
                );

                // Send location data to backend for traffic prediction
                sendLocationToBackend(
                    place.geometry.location.lat(), 
                    place.geometry.location.lng(),
                    duration
                );
            } else {
                coordinatesDisplay.textContent = 'Location not found. Please try again.';
                placeDetailsDisplay.innerHTML = '';
                trafficResultsDisplay.innerHTML = '';
            }
        });
    }

    // Send location data to backend
    function sendLocationToBackend(lat, lng, duration) {
        const trafficResultsDisplay = document.getElementById('trafficResults');
        
        // Prepare form data
        const formData = new FormData();
        formData.append('latitude', lat);
        formData.append('longitude', lng);
        formData.append('duration', duration);

        // Send POST request to backend
        fetch('/traffic_prediction', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Update UI with traffic prediction results
            trafficResultsDisplay.innerHTML = `
                <p><strong>Green Light Duration:</strong> ${data.green_duration} seconds</p>
                <p><strong>Red Light Duration:</strong> ${data.red_duration} seconds</p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            trafficResultsDisplay.innerHTML = 'Error predicting traffic durations';
        });
    }
    
    function resetAll() {
        // Clear input fields
        locationInput.value = '';
        durationInput.value = '';
    
        // Clear location details
        coordinatesElement.textContent = 'Location details will appear here';
    
        // Clear place details
        placeDetailsElement.innerHTML = '<h3 class="section-title">Place Information</h3>';
    
        // Clear traffic results
        trafficResultsElement.innerHTML = '<h3 class="section-title">Traffic Prediction Results</h3>';
    
        // Reset map (if you're using Google Maps)
        if (map) {
            map.setCenter({lat: 0, lng: 0}); // Reset to default center
            map.setZoom(2); // Reset to default zoom
        }
    
        // Remove existing marker if present
        if (marker) {
            marker.setMap(null);
        }
    }

    // Initialize map on page load
    initMap();
    
    // Event listeners
    resetButton.addEventListener('click', resetAll);
    const findLocationBtn = document.getElementById('findLocation');
    findLocationBtn.addEventListener('click', findLocation);

    const locationInput = document.getElementById('locationInput');
    locationInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            findLocation();
        }
    });
});