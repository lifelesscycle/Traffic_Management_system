function initializePage(emergencies) {
    const tableBody = document.getElementById('tableBody');
    const searchInput = document.getElementById('searchInput');

    // Function to populate table
    function populateTable(data) {
        tableBody.innerHTML = ''; // Clear existing rows
        
        Object.entries(data).forEach(([id, emergencys]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${id}</td>
                <td>${emergencys.emergency}</td>
                <td>${emergencys.location}</td>
                <td>${emergencys.city}</td>
                <td>${emergencys.state}</td>
                <td>${emergencys.message}</td>
            `;
            tableBody.appendChild(row);
        });
    }

    // Initial population
    populateTable(emergencies);

    // Search functionality
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const filteredEmergencies = {};

        Object.entries(emergencies).forEach(([id, emergencys]) => {
            const searchString = `${id} ${emergencys.emergency} ${emergencys.location} ${emergencys.city} ${emergencys.state} ${emergencys.message}`.toLowerCase();
            if (searchString.includes(searchTerm)) {
                filteredEmergencies[id] = emergencys;
            }
        });

        populateTable(filteredEmergencies);
    });
}

// Table sorting functionality
function sortTable(columnIndex) {
    const tableBody = document.getElementById('tableBody');
    const rows = Array.from(tableBody.querySelectorAll('tr'));
    const isNumeric = columnIndex === 0; // ID column is numeric

    rows.sort((a, b) => {
        const aColText = a.getElementsByTagName('td')[columnIndex].textContent;
        const bColText = b.getElementsByTagName('td')[columnIndex].textContent;

        if (isNumeric) {
            return aColText.localeCompare(bColText, undefined, {numeric: true});
        } else {
            return aColText.localeCompare(bColText);
        }
    });

    // Reinsert sorted rows
    rows.forEach(row => tableBody.appendChild(row));
}