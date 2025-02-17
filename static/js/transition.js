// static/js/transitions.js
document.addEventListener('DOMContentLoaded', () => {
    const loadingOverlay = document.getElementById('loading-overlay');
    const content = document.getElementById('content');
    const pageLinks = document.querySelectorAll('.page-link');

    // Fade in content and hide loading overlay
    function fadeInContent() {
        setTimeout(() => {
            content.classList.add('loaded');
            loadingOverlay.style.opacity = 0;
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 300);
        }, 100);
    }

    // Page transition on link click
    pageLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const href = link.getAttribute('href');
            
            loadingOverlay.style.display = 'flex';
            loadingOverlay.style.opacity = 1;
            content.style.opacity = 0;

            setTimeout(() => {
                window.location.href = href;
            }, 300);
        });
    });

    // Fade in elements on scroll
    const fadeElements = document.querySelectorAll('.fade-in');
    
    const fadeInObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    fadeElements.forEach(element => {
        fadeInObserver.observe(element);
    });

    // Fade in page content
    fadeInContent();
});