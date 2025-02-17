document.addEventListener('DOMContentLoaded', function() {
  const exploreBtn = document.querySelector('.explore-btn');
  
  if (exploreBtn) {
      exploreBtn.addEventListener('click', function() {
          // Redirect to login page
          window.location.href = "/login";

          // Or if using a direct path
          // window.location.href = "/login";
          
          console.log('Explore More button clicked');
      });
  } else {
      console.error('Explore button not found');
  }
});