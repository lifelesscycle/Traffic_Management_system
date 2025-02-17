const form = document.getElementById('traffic-prediction-form');
const resultContainer = document.getElementById('result-container');
const resultMessage = document.getElementById('result-message');
const addEmergencyButton = document.getElementById('add-emergency');

form.addEventListener('submit', (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const trafficEmergencyData = {
    emergency: formData.get('emergency'),
    location: formData.get('location'),
    city: formData.get('city'),
    state: formData.get('state'),
    message: formData.get('message')
  };
  
  fetch('/emergency', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(trafficEmergencyData)
  })
  .then(response => response.json())
  .then(data => {
    // Display the result
    resultMessage.textContent = `Your traffic emergency data has been saved. You reported an emergency in ${trafficEmergencyData.location}, ${trafficEmergencyData.city}, ${trafficEmergencyData.state}.`;
    resultContainer.classList.remove('hidden');
  })
  .catch(error => {
    console.error('Error:', error);
    resultMessage.textContent = 'An error occurred while saving the emergency.';
    resultContainer.classList.remove('hidden');
  });
});

document.addEventListener('DOMContentLoaded', function() {
  const exploreBtn = document.querySelector('.showButton');
  
  if (exploreBtn) {
      exploreBtn.addEventListener('click', function() {
          window.location.href = "/display_emergency_list";
      });
  }else{
    console.error('Button not found');
  }
  });
  
addEmergencyButton.addEventListener('click', () => {
  form.reset();
  resultContainer.classList.add('hidden');
});