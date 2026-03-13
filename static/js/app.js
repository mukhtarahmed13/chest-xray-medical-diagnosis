// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const previewSection = document.getElementById('previewSection');
const preview = document.getElementById('preview');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const predictionsDiv = document.getElementById('predictions');
const gradcamSection = document.getElementById('gradcamSection');
const gradcamsDiv = document.getElementById('gradcams');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Upload box click handler
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Threshold slider handler
thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = e.target.value;
});

// Analyze button handler
analyzeBtn.addEventListener('click', analyzeImage);

// Handle file selection
function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    if (!file.type.match('image.*')) {
        showError('Please select a valid image file');
        return;
    }

    selectedFile = file;
    analyzeBtn.disabled = false;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) return;

    // Hide previous results and errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'block';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', thresholdSlider.value);

    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to analyze image');
        }

        const data = await response.json();

        // Hide loading
        loadingSection.style.display = 'none';

        // Display results
        displayResults(data);

    } catch (error) {
        loadingSection.style.display = 'none';
        showError('Error analyzing image: ' + error.message);
    }
}

// Display results
function displayResults(data) {
    // Clear previous results
    predictionsDiv.innerHTML = '';
    gradcamsDiv.innerHTML = '';

    // Display predictions
    if (data.predictions && data.predictions.length > 0) {
        data.predictions.forEach(pred => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            
            if (pred.disease === 'No significant findings') {
                predItem.classList.add('no-finding');
            }

            predItem.innerHTML = `
                <span class="disease-name">${pred.disease}</span>
                <span class="probability">${(pred.probability * 100).toFixed(1)}%</span>
            `;
            predictionsDiv.appendChild(predItem);
        });
    }

    // Display Grad-CAM visualizations
    if (data.gradcams && data.gradcams.length > 0) {
        gradcamSection.style.display = 'block';
        
        data.gradcams.forEach(gradcam => {
            const gradcamItem = document.createElement('div');
            gradcamItem.className = 'gradcam-item';
            gradcamItem.innerHTML = `
                <h3>${gradcam.disease}</h3>
                <img src="data:image/png;base64,${gradcam.image}" alt="${gradcam.disease} Grad-CAM">
            `;
            gradcamsDiv.appendChild(gradcamItem);
        });
    } else {
        gradcamSection.style.display = 'none';
    }

    // Show results section
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorSection.style.display = 'none';
    }, 5000);
}

// Initialize
console.log('Chest X-Ray Medical Diagnosis App Loaded');
