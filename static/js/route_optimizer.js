let map;
let directionsService;
let directionsRenderer;

// Initialize the map when the page loads
function initMap() {
    // Create map centered on a default location
    map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 0, lng: 0 },
        zoom: 3
    });

    // Initialize Directions Service and Renderer
    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer();
    directionsRenderer.setMap(map);
}

// Function to calculate and display route
function findRoute() {
    const startLocation = document.getElementById('startLocationInput').value;
    const endLocation = document.getElementById('endLocationInput').value;
    const transportMode = document.getElementById('transportModeInput').value;

    // Geocoding and route calculation
    const geocoder = new google.maps.Geocoder();

    geocoder.geocode({ address: startLocation }, (startResults, startStatus) => {
        if (startStatus === 'OK') {
            geocoder.geocode({ address: endLocation }, (endResults, endStatus) => {
                if (endStatus === 'OK') {
                    const start = startResults[0].geometry.location;
                    const end = endResults[0].geometry.location;

                    // Display coordinates in route info
                    document.getElementById('routeCoordinates').innerHTML = `
                        Start: ${start.lat()}, ${start.lng()}<br>
                        End: ${end.lat()}, ${end.lng()}
                    `;

                    // Request route
                    const request = {
                        origin: start,
                        destination: end,
                        travelMode: google.maps.TravelMode[transportMode]
                    };

                    directionsService.route(request, (result, status) => {
                        if (status === 'OK') {
                            directionsRenderer.setDirections(result);

                            // Calculate and display route estimates
                            const route = result.routes[0];
                            const leg = route.legs[0];
                            
                            document.getElementById('routeEstimates').innerHTML = `
                                <p>Distance: ${leg.distance.text}</p>
                                <p>Duration: ${leg.duration.text}</p>
                            `;
                        } else {
                            alert('Directions request failed due to ' + status);
                        }
                    });
                } else {
                    alert('Geocode was not successful for the end location');
                }
            });
        } else {
            alert('Geocode was not successful for the start location');
        }
    });
}

// Reset function
function resetRoute() {
    // Clear map and inputs
    directionsRenderer.setMap(null);
    initMap();
    
    document.getElementById('startLocationInput').value = '';
    document.getElementById('endLocationInput').value = '';
    document.getElementById('routeCoordinates').innerHTML = 'Route details will appear here';
    document.getElementById('routeEstimates').innerHTML = '';
}

// Add event listeners when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize map
    initMap();

    // Add event listeners
    document.getElementById('findRouteButton').addEventListener('click', findRoute);
    document.getElementById('resetButton').addEventListener('click', resetRoute);
});

// Error handling for Google Maps API
window.gm_authFailure = () => {
    alert('Google Maps authentication failed. Please check your API key.');
};