document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const fakeNewsForm = document.getElementById('fakeNewsForm');
    const fileInput = document.getElementById('image');
    const fileNameDisplay = document.getElementById('file-name');
    const loadingIndicator = document.getElementById('loading');
    const resultContainer = document.getElementById('result');
    const toast = document.getElementById('toast');

    // Ensure loading indicator is hidden initially
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }

    // Toggle Navigation Menu
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', !isExpanded);
            navMenu.classList.toggle('open');
        });
    }

    // File Input Handler
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = `Selected file: ${fileInput.files[0].name}`;
            } else {
                fileNameDisplay.textContent = '';
            }
        });
    }

    // Form Submit Handler
    if (fakeNewsForm) {
        fakeNewsForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            // Get form data
            const newsText = document.getElementById('news_text').value.trim();
            const imageFile = fileInput.files[0];

            // Validate input
            if (newsText.length < 50) {
                showToast('Please enter at least 50 characters of news text.', 'error');
                return;
            }

            if (!imageFile) {
                showToast('Please upload an image related to the news.', 'error');
                return;
            }

            try {
                // Show loading indicator
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'flex';
                }

                // Prepare form data
                const formData = new FormData();
                formData.append('news_text', newsText);
                formData.append('image', imageFile);

                // Send request to backend
                const response = await fetch('/detect_fake_news', {
                    method: 'POST',
                    body: formData
                });


                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                console.log(data);
                
                // Hide loading indicator
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }

                // Display results
                displayResults(data);
                
            } catch (error) {
                console.error('Error:', error);
                // Hide loading indicator
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                showToast('An error occurred while analyzing the news. Please try again.', 'error');
            }
        });
    }


// Get Modal Elements
const factCheckModal = document.getElementById('factCheckModal');
const closeModalBtn = document.getElementById('closeModal');

// Event Listener to Close Modal
if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => {
        closeModal();
    });
}

// Function to Open Modal
function openModal() {
    if (factCheckModal) {
        factCheckModal.style.display = 'block';
        factCheckModal.setAttribute('aria-hidden', 'false');
    }
}

// Function to Close Modal
function closeModal() {
    if (factCheckModal) {
        factCheckModal.style.display = 'none';
        factCheckModal.setAttribute('aria-hidden', 'true');
    }
}

// Close Modal When Clicking Outside of Modal Content
window.addEventListener('click', (event) => {
    if (event.target === factCheckModal) {
        closeModal();
    }
});



function displayResults(data) {
    if (!resultContainer) return;

    resultContainer.innerHTML = '';
    resultContainer.classList.remove('result-success', 'result-error', 'result-warning');
    
    // Determine result class based on fake_news status
    const resultClass = data.fake_news === 'Yes' ? 'result-error' : 
                       data.fake_news === 'No' ? 'result-success' : 
                       'result-warning'; // For 'Inconclusive'
    resultContainer.classList.add(resultClass);

    const resultHTML = `
        <div class="result-header">
            <h3>Analysis Results</h3>
            <div class="confidence-score">
                <span>Confidence: ${(data.fake_news_confidence * 100).toFixed(1)}%</span>
            </div>
        </div>

        <div class="result-summary">
            <div class="status-badge ${data.fake_news.toLowerCase()}">
                Status: ${data.fake_news}
            </div>
            <p class="reasoning">${data.reasoning}</p>
        </div>

        <div class="analysis-details">
            <div class="detail-section">
                <h4>Image-Text Analysis</h4>
                <p>Match: ${data.image_text_match}</p>
                <p>Confidence: ${(data.image_text_match_confidence * 100).toFixed(1)}%</p>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: ${data.image_text_match_confidence * 100}%;"></div>
                </div>
            </div>

            <div class="detail-section">
                <h4>Image Analysis</h4>
                <p>${data.image_analysis.description}</p>
            </div>

            <div class="detail-section">
                <h4>Fact Check Results Summary</h4>
                <p>Status: ${data.fact_check}</p>
                <p>Confidence: ${(data.fact_check_confidence * 100).toFixed(1)}%</p>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: ${data.fact_check_confidence * 100}%;"></div>
                </div>
                <button class="view-factcheck-btn" aria-label="View Fact-Check Details">View Fact-Check Details</button>
            </div>
        </div>
    `;

    resultContainer.innerHTML = resultHTML;
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth' });

    // Add Event Listener to "View Fact-Check Details" Button
    const viewFactCheckBtn = resultContainer.querySelector('.view-factcheck-btn');
    if (viewFactCheckBtn) {
        viewFactCheckBtn.addEventListener('click', () => {
            // Store fact-check results in localStorage
            localStorage.setItem('factCheckResults', JSON.stringify(data.fact_check_results));
            // Navigate to fact-check details page
            window.location.href = 'factcheck.html';
        });
    }
}

// Function to Populate Fact-Check Details in Modal
function populateFactCheckDetails(factCheckResults) {
    if (!factCheckDetails) return;

    // Create HTML for Fact-Check Results
    const factCheckHTML = `
        <ul>
            ${factCheckResults.map(result => `
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

    // Inject HTML into Modal
    factCheckDetails.innerHTML = factCheckHTML;
}

    // Toast Notification Function
    function showToast(message, type = 'success') {
        if (!toast) return;

        toast.textContent = message;
        toast.style.borderLeftColor = type === 'success' ? 'var(--success-color)' : 'var(--error-color)';
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
});
