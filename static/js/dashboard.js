// Function to create circular progress charts
function createCircularChart(canvasId, percentage, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [color, '#2a3547'],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '80%',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Function to create line chart
function createLineChart() {
    const lineCtx = document.getElementById('lineChart').getContext('2d');
    return new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: ['8:00', '10:00', '12:00', '14:00', '16:00', '18:00'],
            datasets: [{
                label: 'Progress',
                data: [65, 59, 80, 81, 56, 55],
                borderColor: '#ffd700',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#2a3547'
                    },
                    ticks: {
                        color: '#8a94a6'
                    }
                },
                x: {
                    grid: {
                        color: '#2a3547'
                    },
                    ticks: {
                        color: '#8a94a6'
                    }
                }
            }
        }
    });
}

// Function to create progress chart
function createProgressChart() {
    const progressCtx = document.getElementById('progressChart').getContext('2d');
    return new Chart(progressCtx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            datasets: [{
                data: [30, 45, 60, 75, 90],
                borderColor: '#4cc9f0',
                backgroundColor: 'rgba(76, 201, 240, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#2a3547'
                    },
                    ticks: {
                        color: '#8a94a6'
                    }
                },
                x: {
                    grid: {
                        color: '#2a3547'
                    },
                    ticks: {
                        color: '#8a94a6'
                    }
                }
            }
        }
    });
}

// Initialize all charts
document.addEventListener('DOMContentLoaded', () => {
    // Create circular progress charts
    createCircularChart('designChart', 74, '#ffd700');
    createCircularChart('frontendChart', 48, '#ff6b6b');
    createCircularChart('backendChart', 62, '#4cc9f0');
    createCircularChart('testingChart', 85, '#2ecc71');

    // Create line chart
    createLineChart();

    // Create progress chart
    createProgressChart();
});