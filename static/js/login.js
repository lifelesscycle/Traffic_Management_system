document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const usernameError = document.getElementById('username-error');
    const passwordError = document.getElementById('password-error');

    // Function to validate username
    function validateUsername(username) {
        // Basic validation - username should not be empty and meet certain criteria
        if (username.trim() === '') {
            usernameError.textContent = 'Username cannot be empty';
            return false;
        }
        
        // Optional: Add more specific username validation
        if (username.length < 3) {
            usernameError.textContent = 'Username must be at least 3 characters long';
            return false;
        }
        
        usernameError.textContent = '';
        return true;
    }

    // Function to validate password
    function validatePassword(password) {
        // Basic validation - password should not be empty and meet certain criteria
        if (password.trim() === '') {
            passwordError.textContent = 'Password cannot be empty';
            return false;
        }
        
        // Optional: Add more specific password validation
        if (password.length < 6) {
            passwordError.textContent = 'Password must be at least 6 characters long';
            return false;
        }
        
        passwordError.textContent = '';
        return true;
    }

    // Event listeners for real-time validation
    usernameInput.addEventListener('input', function() {
        validateUsername(this.value);
    });

    passwordInput.addEventListener('input', function() {
        validatePassword(this.value);
    });

    // Form submission handler
    loginForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission

        // Clear previous error messages
        usernameError.textContent = '';
        passwordError.textContent = '';

        // Get input values
        const username = usernameInput.value.trim();
        const password = passwordInput.value.trim();

        // Validate inputs
        const isUsernameValid = validateUsername(username);
        const isPasswordValid = validatePassword(password);

        // If both inputs are valid, proceed with login
        if (isUsernameValid && isPasswordValid) {
            // Perform login request
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Redirect to dashboard or home page
                    window.location.href = '/dashboard';
                } else {
                    // Show error message from server
                    if (data.error === 'username') {
                        usernameError.textContent = 'Invalid username';
                    } else if (data.error === 'password') {
                        passwordError.textContent = 'Incorrect password';
                    } else {
                        // Generic error handling
                        alert('Login failed. Please try again.');
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            });
        }
    });
});