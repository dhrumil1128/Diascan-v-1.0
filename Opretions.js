const images = [
  "img/img-10.jpg",
  "img/im-9.jpg",
  "img/img-7.jpg"
];

let currentIndex = 0;
const backgroundContainer = document.getElementById('background-container');

// Function to change the background image
function changeBackgroundImage() {
  currentIndex = (currentIndex + 1) % images.length;
  backgroundContainer.style.backgroundImage = `url(${images[currentIndex]})`;
}

// Change background every 5 seconds
setInterval(changeBackgroundImage, 5000);

// Set the initial background image
backgroundContainer.style.backgroundImage = `url(${images[0]})`;

// Wait for the DOM to fully load before running the script
document.addEventListener('DOMContentLoaded', function() {
  var predictButton = document.getElementById('predictButton');
  predictButton.addEventListener('click', function() {
      // Get input values
      var age = parseInt(document.getElementById('age').value);
      var glucose = parseInt(document.getElementById('glucose').value);
      var bloodPressure = parseInt(document.getElementById('bloodPressure').value);
      var insulin = parseInt(document.getElementById('insulin').value);
      var bmi = parseFloat(document.getElementById('bmi').value);
      var diabetesPedigree = parseFloat(document.getElementById('diabetesPedigree').value);

      // Validate form inputs
      if (isNaN(age) || isNaN(glucose) || isNaN(bloodPressure) || isNaN(insulin) || isNaN(bmi) || isNaN(diabetesPedigree)) {
          alert('Please enter valid data in all fields.');
          return;
      }

      // Prepare data to send as JSON
      var data = {
          age: age,
          glucose: glucose,
          bloodPressure: bloodPressure,
          insulin: insulin,
          bmi: bmi,
          diabetesPedigree: diabetesPedigree
      };

      // Send POST request to Flask API endpoint
      fetch('http://127.0.0.1:8000/predict', { // Ensure the URL matches your Flask endpoint
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify(data)
      })
      .then(response => response.json())
      .then(data => {
          // Handle the prediction result
          alert('Prediction Result: ' + data.prediction);
      })
      .catch(error => {
          console.error('Error:', error);
          alert('Prediction failed. Please try again.');
      });
  });

  // Reset button functionality
  var resetButton = document.getElementById('resetButton');
  resetButton.addEventListener('click', function() {
      document.getElementById('age').value = '';
      document.getElementById('glucose').value = '';
      document.getElementById('bloodPressure').value = '';
      document.getElementById('insulin').value = '';
      document.getElementById('bmi').value = '';
      document.getElementById('diabetesPedigree').value = '';
  });
});
