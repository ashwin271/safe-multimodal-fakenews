document.addEventListener('DOMContentLoaded', () => {
    const factCheckDetails = document.getElementById('factCheckDetails');

    // Retrieve fact-check results from localStorage
    const factCheckData = localStorage.getItem('factCheckResults');
    if (factCheckData) {
        const parsedData = JSON.parse(factCheckData);

        // Create HTML for Fact-Check Results
        const factCheckHTML = `
            <ul>
                ${parsedData.map(result => `
                    <li>
                        <a href="${result.url}" target="_blank" rel="noopener noreferrer">
                            ${result.title}
                        </a>
                        <p class="source-content">${result.content}</p>
                        <div class="source-score">
                            Relevance: ${(result.score * 100).toFixed(1)}%
                        </div>
                    </li>
                `).join('')}
            </ul>
        `;

        // Inject HTML into Fact-Check Details Section
        factCheckDetails.innerHTML = factCheckHTML;
    } else {
        factCheckDetails.innerHTML = '<p>No fact-check data available.</p>';
    }
});