document.addEventListener('DOMContentLoaded', function() {
  let map;
  let markers = [];
  let locationDict = {};

  // Initialize Google Maps with more specific options
  function initMap() {
      map = new google.maps.Map(document.getElementById('map'), {
          center: { lat: 40.730610, lng: -73.935242 }, // Set the initial center to New York City 
          zoom: 8,
          mapTypeControl: true,
          streetViewControl: false,
          fullscreenControl: true
      });

      // Add click event to add markers
      map.addListener('click', function(event) {
          addMarker(event.latLng);
      });
  }

  // Add marker to the map with enhanced functionality
  function addMarker(location) {
      // Limit to one marker per location name
      const locationName = prompt('Enter location name:');
      if (!locationName) return;

      // Remove any existing marker for this location
      markers = markers.filter(marker => {
          if (marker.getTitle() === locationName) {
              marker.setMap(null);
              return false;
          }
          return true;
      });

      const marker = new google.maps.Marker({
          position: location,
          map: map,
          title: locationName,
          animation: google.maps.Animation.DROP,
          draggable: true
      });

      // Add info window
      const infoWindow = new google.maps.InfoWindow({
          content: `<strong>${locationName}</strong><br>Lat: ${location.lat()}<br>Lng: ${location.lng()}`
      });

      marker.addListener('click', () => {
          infoWindow.open(map, marker);
      });

      markers.push(marker);
      updateLocationDict(locationName, location);
  }

  // Handle form submission with enhanced validation
  document.getElementById('add-location-form').addEventListener('submit', function(event) {
      event.preventDefault();

      const city = document.getElementById('city').value.trim();
      const locationsInput = document.getElementById('locations').value.trim();
      
      // Basic validation
      if (!city) {
          alert('Please enter a city name');
          return;
      }

      // Split locations and create dictionary
      const locations = locationsInput.split(',')
          .map(loc => loc.trim())
          .filter(loc => loc !== ''); // Remove empty entries
      
      if (locations.length === 0) {
          alert('Please enter at least one location');
          return;
      }

      // Create location dictionary
      locationDict = {
          city: city,
          locations: {}
      };

      locations.forEach((loc) => {
          locationDict.locations[loc] = null; // Default to null if no marker
      });

      // Send data to server
      sendLocationData(locationDict);
  });

  // Update location dictionary with marker coordinates
  function updateLocationDict(locationName, location) {
      if (!locationDict.locations) {
          locationDict.locations = {};
      }
      locationDict.locations[locationName] = {
          lat: location.lat(),
          lng: location.lng()
      };
  }

  // Send location data to Flask backend
  function sendLocationData(data) {
      fetch('/add-location', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(data)
      })
      .then(response => {
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          return response.json();
      })
      .then(result => {
          // Display result
          const resultContainer = document.getElementById('result-container');
          resultContainer.innerHTML = `
              <div class="success-message">
                  <p>City: ${result.city}</p>
                  <p>Locations Added: ${Object.keys(result.locations).join(', ')}</p>
              </div>
          `;
          // Clear form after successful submission
          document.getElementById('city').value = '';
          document.getElementById('locations').value = '';
      })
      .catch(error => {
          console.error('Error:', error);
          const resultContainer = document.getElementById('result-container');
          resultContainer.innerHTML = `
              <div class="error-message">
                  <p>Error adding locations</p>
                  <p>${error.message}</p>
              </div>
          `;
      });
  }

  // Initialize map when DOM is ready
  initMap();
});
