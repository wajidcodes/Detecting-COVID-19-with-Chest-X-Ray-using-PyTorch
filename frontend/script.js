document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    let currentFile = null;
    
    // API URL (adjusts automatically based on where it's hosted, usually reverse-proxied to /api/predict via Nginx)
    // If testing locally without Docker, point directly to FastAPI (http://localhost:8000/predict)
    const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8000/predict' : '/api/predict';

    // File Upload Handlers
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
        dropZone.style.backgroundColor = '#eff6ff';
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.backgroundColor = 'transparent';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
        dropZone.style.backgroundColor = 'transparent';
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file (PNG/JPG)');
            return;
        }
        currentFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
    
    clearBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsSection.classList.add('hidden');
    });
    
    // Analyze Button Handler
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        // UI State update
        previewContainer.classList.add('hidden');
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('API Error');
            
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            
            displayResults(data);
            
        } catch (error) {
            alert('Error analyzing image. Ensure backend is running.');
            console.error(error);
            previewContainer.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });
    
    function displayResults(data) {
        resultsSection.classList.remove('hidden');
        previewContainer.classList.remove('hidden'); // Show image again above results
        analyzeBtn.classList.add('hidden'); // Hide analyze button once analyzed
        
        // 1. Update Primary Prediction
        const badge = document.getElementById('prediction-badge');
        badge.textContent = data.predicted_class.replace('_', ' ');
        
        // Reset classes
        badge.className = 'prediction-badge';
        
        // Apply color based on class
        if (data.predicted_class === 'COVID') badge.classList.add('badge-COVID');
        else if (data.predicted_class === 'Normal') badge.classList.add('badge-Normal');
        else if (data.predicted_class === 'Lung_Opacity') badge.classList.add('badge-Lung_Opacity');
        else badge.classList.add('badge-Viral');
        
        document.getElementById('confidence-value').textContent = data.confidence;
        
        // 2. Update Probability Bars
        const probs = data.probabilities;
        
        // COVID
        document.getElementById('bar-covid').style.width = probs['COVID'] + '%';
        document.getElementById('pct-covid').textContent = probs['COVID'] + '%';
        
        // Normal
        document.getElementById('bar-normal').style.width = probs['Normal'] + '%';
        document.getElementById('pct-normal').textContent = probs['Normal'] + '%';
        
        // Lung Opacity
        document.getElementById('bar-opacity').style.width = probs['Lung_Opacity'] + '%';
        document.getElementById('pct-opacity').textContent = probs['Lung_Opacity'] + '%';
        
        // Viral Pneumonia (handle key formatting mismatch if any)
        const viralProb = probs['Viral Pneumonia'] || probs['Viral_Pneumonia'] || 0;
        document.getElementById('bar-viral').style.width = viralProb + '%';
        document.getElementById('pct-viral').textContent = viralProb + '%';
    }
    
    // Check backend connection on load
    fetch(API_URL.replace('/predict', '/'))
        .then(res => {
            if (res.ok) {
                document.getElementById('status-badge').textContent = 'Backend Connected';
                document.getElementById('status-badge').className = 'status connected';
            }
        })
        .catch(err => {
            document.getElementById('status-badge').textContent = 'Backend Offline';
            document.getElementById('status-badge').className = 'status error';
        });
});
