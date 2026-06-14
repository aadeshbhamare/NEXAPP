/* ==========================================
   NEXUS AI STUDIO - MASTER LOGIC CONTROLLER
   ========================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // Setup Global Navigation & Routing
    initNavigation();

    // Initialize Sub-Modules
    initToast();
    initNeuralNetworkLab();
    initVectorSpaceLab();
    initTokenizerLab();
    initClubPortal();
    initQuizArena();
    initAIChatConsole();
});

/* ==========================================
   GLOBAL UTILITIES
   ========================================== */

// Global Toast Banner
let showToast = (msg) => {};
function initToast() {
    const toast = document.getElementById('toast-banner');
    const toastText = document.getElementById('toast-text');
    
    showToast = (msg) => {
        toastText.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    };
}

// Sidebar & Page Routing
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const sidebar = document.getElementById('app-sidebar');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active status from nav items
            navItems.forEach(n => n.classList.remove('active'));
            // Add active status to clicked item
            item.classList.add('active');

            // Toggle active tabs
            const targetTab = item.getAttribute('data-tab');
            tabContents.forEach(tab => {
                if (tab.id === `tab-${targetTab}`) {
                    tab.classList.add('active-tab');
                } else {
                    tab.classList.remove('active-tab');
                }
            });

            // Close sidebar on mobile
            sidebar.classList.remove('mobile-open');
            mobileMenuToggle.innerHTML = '<i data-lucide="menu"></i>';
            lucide.createIcons();

            // Notify user
            showToast(`Navigated to ${item.querySelector('span').textContent}`);
        });
    });

    // Mobile Menu Toggle
    mobileMenuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('mobile-open');
        if (sidebar.classList.contains('mobile-open')) {
            mobileMenuToggle.innerHTML = '<i data-lucide="x"></i>';
        } else {
            mobileMenuToggle.innerHTML = '<i data-lucide="menu"></i>';
        }
        lucide.createIcons();
    });

    // Check URL parameters for initial tab load
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');
    if (tabParam) {
        const targetNavItem = document.querySelector(`.nav-item[data-tab="${tabParam}"]`);
        if (targetNavItem) {
            setTimeout(() => {
                targetNavItem.click();
            }, 100);
        }
    }
}


/* ==========================================
   TAB 2: NEURAL NETWORK PLAYGROUND
   ========================================== */
function initNeuralNetworkLab() {
    const canvas = document.getElementById('boundary-canvas');
    const ctx = canvas.getContext('2d');
    const topologySvg = document.getElementById('topology-svg');

    // UI Controls
    const datasetSelect = document.getElementById('nn-dataset-select');
    const lrSelect = document.getElementById('nn-lr-select');
    const activationSelect = document.getElementById('nn-activation-select');
    const layersSlider = document.getElementById('nn-layers-slider');
    const layersVal = document.getElementById('nn-layers-val');
    const neuronsSlider = document.getElementById('nn-neurons-slider');
    const neuronsVal = document.getElementById('nn-neurons-val');
    
    const trainBtn = document.getElementById('nn-train-btn');
    const trainLbl = document.getElementById('nn-train-lbl');
    const resetBtn = document.getElementById('nn-reset-btn');
    const epochCountLbl = document.getElementById('nn-epoch-count');
    const lossValLbl = document.getElementById('nn-loss-val');

    // Neural Network State
    let nn = null;
    let dataset = [];
    let trainingInterval = null;
    let epoch = 0;
    let isTraining = false;

    // Activation Functions
    const activations = {
        sigmoid: {
            f: x => 1 / (1 + Math.exp(-x)),
            df: y => y * (1 - y)
        },
        tanh: {
            f: x => Math.tanh(x),
            df: y => 1 - y * y
        },
        relu: {
            f: x => Math.max(0, x),
            df: y => y > 0 ? 1 : 0
        }
    };

    // Feedforward Neural Network Class
    class SimpleNeuralNetwork {
        constructor(inputDim, hiddenSizes, outputDim, actName) {
            this.actName = actName;
            this.act = activations[actName];
            this.layers = []; // Weights matrices & Bias vectors
            
            // Build sizes array: [input, hidden1, ..., output]
            const sizes = [inputDim, ...hiddenSizes, outputDim];
            
            for (let i = 0; i < sizes.length - 1; i++) {
                const rows = sizes[i + 1];
                const cols = sizes[i];
                
                // Xavier/He-like Random Initialization
                const weights = Array.from({length: rows}, () => 
                    Array.from({length: cols}, () => (Math.random() - 0.5) * 2 * Math.sqrt(2 / cols))
                );
                const biases = Array.from({length: rows}, () => 0.0);
                this.layers.push({ weights, biases });
            }
        }

        forward(x) {
            let activationsList = [x];
            let zList = [];
            
            for (let i = 0; i < this.layers.length; i++) {
                const { weights, biases } = this.layers[i];
                const prevA = activationsList[activationsList.length - 1];
                const currentA = [];
                const currentZ = [];
                
                for (let r = 0; r < weights.length; r++) {
                    let sum = biases[r];
                    for (let c = 0; c < weights[r].length; c++) {
                        sum += weights[r][c] * prevA[c];
                    }
                    currentZ.push(sum);
                    // Output layer uses sigmoid for binary classification probability
                    if (i === this.layers.length - 1) {
                        currentA.push(1 / (1 + Math.exp(-sum))); // Sigmoid output
                    } else {
                        currentA.push(this.act.f(sum));
                    }
                }
                zList.push(currentZ);
                activationsList.push(currentA);
            }
            return { activationsList, zList };
        }

        backprop(x, target, lr) {
            const { activationsList } = this.forward(x);
            const output = activationsList[activationsList.length - 1][0];
            
            // Output Layer Error (Mean Squared Error gradient with respect to output activation)
            let delta = (output - target) * (output * (1 - output)); // Sigmoid derivative output
            
            // Backpropagate deltas
            let deltas = [delta];
            for (let i = this.layers.length - 2; i >= 0; i--) {
                const nextDelta = deltas[0];
                const nextWeights = this.layers[i + 1].weights;
                const currentA = activationsList[i + 1];
                const currentDeltas = [];
                
                for (let c = 0; c < nextWeights[0].length; c++) {
                    let err = 0;
                    for (let r = 0; r < nextWeights.length; r++) {
                        err += nextWeights[r][c] * nextDelta;
                    }
                    // Hidden layer derivative
                    currentDeltas.push(err * this.act.df(currentA[c]));
                }
                deltas.unshift(currentDeltas);
            }

            // Update Weights & Biases
            for (let i = 0; i < this.layers.length; i++) {
                const { weights, biases } = this.layers[i];
                const layerDelta = deltas[i];
                const layerInput = activationsList[i];
                
                for (let r = 0; r < weights.length; r++) {
                    biases[r] -= lr * layerDelta[r];
                    for (let c = 0; c < weights[r].length; c++) {
                        weights[r][c] -= lr * layerDelta[r] * layerInput[c];
                    }
                }
            }
            // Return Squared Error
            return 0.5 * (output - target) ** 2;
        }
    }

    // Dataset Generators
    function generateDataset(type) {
        const data = [];
        const size = 150;
        
        for (let i = 0; i < size; i++) {
            const x = (Math.random() - 0.5) * 2; // -1 to 1
            const y = (Math.random() - 0.5) * 2;
            let label = 0;
            
            if (type === 'linear') {
                label = (x + y > 0) ? 1 : 0;
            } else if (type === 'xor') {
                label = (x * y > 0) ? 1 : 0;
            } else if (type === 'circle') {
                label = (x*x + y*y < 0.4) ? 1 : 0;
            }
            data.push({ x: [x, y], y: label });
        }
        return data;
    }

    // Initialize Network State
    function rebuildNetwork() {
        epoch = 0;
        lossValLbl.textContent = "0.000";
        epochCountLbl.textContent = "0000";
        
        const numLayers = parseInt(layersSlider.value);
        const numNeurons = parseInt(neuronsSlider.value);
        const hiddenLayers = Array.from({length: numLayers}, () => numNeurons);
        const actFunc = activationSelect.value;
        
        dataset = generateDataset(datasetSelect.value);
        nn = new SimpleNeuralNetwork(2, hiddenLayers, 1, actFunc);
        
        drawDecisionBoundary();
        drawTopologySVG();
    }

    // Render 2D Decision boundary grid
    function drawDecisionBoundary() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Render background predictions
        const resolution = 4; // grid block size
        for (let x = 0; x < canvas.width; x += resolution) {
            for (let y = 0; y < canvas.height; y += resolution) {
                // Map screen coordinate to [-1, 1]
                const nx = (x / canvas.width) * 2 - 1;
                const ny = 1 - (y / canvas.height) * 2; // Invert y-axis
                
                const { activationsList } = nn.forward([nx, ny]);
                const prob = activationsList[activationsList.length - 1][0];
                
                // Color interpolation: blue (class 0) to orange (class 1)
                const r = Math.floor(prob * 234 + (1 - prob) * 14);
                const g = Math.floor(prob * 88 + (1 - prob) * 165);
                const b = Math.floor(prob * 12 + (1 - prob) * 233);
                
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.25)`;
                ctx.fillRect(x, y, resolution, resolution);
            }
        }

        // Render data points
        dataset.forEach(pt => {
            const sx = (pt.x[0] + 1) / 2 * canvas.width;
            const sy = (1 - pt.x[1]) / 2 * canvas.height;
            
            ctx.beginPath();
            ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
            ctx.fillStyle = pt.y === 1 ? 'rgb(234, 88, 12)' : 'rgb(14, 165, 233)'; // Orange vs Blue
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = 'white';
            ctx.fill();
            ctx.stroke();
        });
    }

    // Draw SVG Topology diagram representing network weights
    function drawTopologySVG() {
        topologySvg.innerHTML = '';
        const width = topologySvg.clientWidth || 350;
        const height = topologySvg.clientHeight || 250;
        
        const padding = 30;
        const layerCount = nn.layers.length + 1;
        const layerXSpacing = (width - 2 * padding) / (layerCount - 1);
        
        // Assemble nodes coordinate layouts
        const nodesByLayer = [];
        
        // Input Layer
        const inputsCount = 2;
        nodesByLayer.push(Array.from({length: inputsCount}, (_, i) => ({
            x: padding,
            y: padding + i * (height - 2 * padding) / (inputsCount - 1),
            activation: 0
        })));
        
        // Hidden Layers
        for (let i = 0; i < nn.layers.length - 1; i++) {
            const hCount = nn.layers[i].weights.length;
            nodesByLayer.push(Array.from({length: hCount}, (_, j) => ({
                x: padding + (i + 1) * layerXSpacing,
                y: hCount === 1 ? height/2 : padding + j * (height - 2 * padding) / (hCount - 1),
                activation: 0
            })));
        }
        
        // Output Layer
        nodesByLayer.push([{
            x: width - padding,
            y: height / 2,
            activation: 0
        }]);

        // Draw connections (weight lines) first
        for (let l = 0; l < nn.layers.length; l++) {
            const { weights } = nn.layers[l];
            const srcNodes = nodesByLayer[l];
            const destNodes = nodesByLayer[l + 1];
            
            for (let d = 0; d < destNodes.length; d++) {
                for (let s = 0; s < srcNodes.length; s++) {
                    const w = weights[d][s];
                    const weightColor = w > 0 ? 'rgba(14, 165, 233, ' : 'rgba(234, 88, 12, '; // Blue vs Orange
                    const thickness = Math.min(5, Math.abs(w) * 1.5);
                    
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', srcNodes[s].x);
                    line.setAttribute('y1', srcNodes[s].y);
                    line.setAttribute('x2', destNodes[d].x);
                    line.setAttribute('y2', destNodes[d].y);
                    line.setAttribute('stroke', `${weightColor}${Math.min(1.0, Math.abs(w))})`);
                    line.setAttribute('stroke-width', Math.max(0.5, thickness));
                    topologySvg.appendChild(line);
                }
            }
        }

        // Draw node circles
        for (let l = 0; l < nodesByLayer.length; l++) {
            const nodes = nodesByLayer[l];
            nodes.forEach(node => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', node.x);
                circle.setAttribute('cy', node.y);
                circle.setAttribute('r', 8);
                circle.setAttribute('fill', '#1e293b');
                circle.setAttribute('stroke', '#94a3b8');
                circle.setAttribute('stroke-width', 2);
                topologySvg.appendChild(circle);
            });
        }
    }

    // Training Tick Loop
    function trainStep() {
        if (!isTraining) return;
        
        const lr = parseFloat(lrSelect.value);
        let errorSum = 0;
        
        // Execute 10 training mini-steps per animation frame for speed
        const stepsPerFrame = 15;
        for (let s = 0; s < stepsPerFrame; s++) {
            dataset.forEach(pt => {
                errorSum += nn.backprop(pt.x, pt.y, lr);
            });
            epoch++;
        }
        
        const averageLoss = errorSum / (dataset.length * stepsPerFrame);
        
        // Update labels
        lossValLbl.textContent = averageLoss.toFixed(4);
        epochCountLbl.textContent = String(epoch).padStart(4, '0');
        
        drawDecisionBoundary();
        drawTopologySVG();
        
        trainingInterval = requestAnimationFrame(trainStep);
    }

    // Event Listeners
    trainBtn.addEventListener('click', () => {
        isTraining = !isTraining;
        if (isTraining) {
            trainBtn.classList.add('btn-active');
            trainLbl.textContent = "Pause";
            trainBtn.querySelector('i').setAttribute('data-lucide', 'pause');
            lucide.createIcons();
            trainStep();
        } else {
            trainBtn.classList.remove('btn-active');
            trainLbl.textContent = "Resume";
            trainBtn.querySelector('i').setAttribute('data-lucide', 'play');
            lucide.createIcons();
            cancelAnimationFrame(trainingInterval);
        }
    });

    resetBtn.addEventListener('click', rebuildNetwork);
    
    // Sliders
    layersSlider.addEventListener('input', (e) => {
        layersVal.textContent = e.target.value;
        rebuildNetwork();
    });
    neuronsSlider.addEventListener('input', (e) => {
        neuronsVal.textContent = e.target.value;
        rebuildNetwork();
    });
    
    // Dropdowns
    datasetSelect.addEventListener('change', rebuildNetwork);
    activationSelect.addEventListener('change', rebuildNetwork);

    // Initial Trigger
    rebuildNetwork();
}


/* ==========================================
   TAB 3: SEMANTIC VECTOR SPACE
   ========================================== */
function initVectorSpaceLab() {
    const vectorSvg = document.getElementById('vector-svg');
    const resultsContainer = document.getElementById('vector-results');
    const queryInput = document.getElementById('vector-query-input');
    const searchBtn = document.getElementById('vector-search-btn');

    // List of preloaded sentences
    const inputFields = document.querySelectorAll('.corpus-input');

    function executeSearch() {
        const sentences = Array.from(inputFields).map(input => input.value.trim()).filter(s => s !== '');
        const query = queryInput.value.trim().toLowerCase();
        
        if (sentences.length === 0 || query === '') {
            showToast("Please supply sentences and a search query.");
            return;
        }

        // 1. Vocabulary Extraction
        const stopWords = new Set(['i', 'a', 'the', 'is', 'are', 'and', 'to', 'in', 'on', 'at', 'every', 'he', 'by']);
        const getWords = (txt) => txt.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 1 && !stopWords.has(w));
        
        let vocab = new Set();
        sentences.forEach(s => getWords(s).forEach(w => vocab.add(w)));
        getWords(query).forEach(w => vocab.add(w));
        vocab = Array.from(vocab);

        if (vocab.length === 0) {
            vocab = ["sample", "word"];
        }

        // 2. Term-Frequency Vectorizer (Simulating Embeddings)
        const vectorize = (words) => {
            const vec = Array.from({length: vocab.length}, () => 0);
            words.forEach(w => {
                const idx = vocab.indexOf(w);
                if (idx !== -1) vec[idx] += 1;
            });
            // L2 Normalization
            const norm = Math.sqrt(vec.reduce((sum, v) => sum + v*v, 0));
            return norm > 0 ? vec.map(v => v / norm) : vec;
        };

        const sentenceVectors = sentences.map(s => vectorize(getWords(s)));
        const queryVector = vectorize(getWords(query));

        // 3. Cosine Similarity Calculation
        const dotProduct = (v1, v2) => v1.reduce((sum, v, i) => sum + v * v2[i], 0);
        
        const similarities = sentences.map((s, idx) => {
            const score = dotProduct(sentenceVectors[idx], queryVector);
            return { text: s, score: score, vector: sentenceVectors[idx] };
        });

        // Sort by similarity score descending
        const sortedResults = [...similarities].sort((a, b) => b.score - a.score);

        // Render ranked results
        resultsContainer.innerHTML = '';
        sortedResults.forEach(res => {
            const item = document.createElement('div');
            item.className = 'vector-result-item';
            item.innerHTML = `
                <span class="vector-result-text">${res.text}</span>
                <span class="vector-result-score">${res.score.toFixed(3)}</span>
            `;
            resultsContainer.innerHTML += item.outerHTML;
        });

        // 4. Dimensionality Reduction to 2D (PCA Approximation in JS)
        // Center the vectors
        const allVectors = [...sentenceVectors, queryVector];
        const mean = Array.from({length: vocab.length}, () => 0);
        allVectors.forEach(v => {
            for (let i = 0; i < vocab.length; i++) mean[i] += v[i];
        });
        for (let i = 0; i < vocab.length; i++) mean[i] /= allVectors.length;

        const centered = allVectors.map(v => v.map((x, i) => x - mean[i]));

        // Simple Power Iteration PCA to find two orthogonal principal directions
        const powerIteration = (data, iterations = 10) => {
            let u = Array.from({length: vocab.length}, () => Math.random() - 0.5);
            for (let iter = 0; iter < iterations; iter++) {
                let nextU = Array.from({length: vocab.length}, () => 0);
                // Covariance approximation
                for (let d = 0; d < data.length; d++) {
                    const doc = data[d];
                    const projection = dotProduct(doc, u);
                    for (let i = 0; i < vocab.length; i++) {
                        nextU[i] += doc[i] * projection;
                    }
                }
                const norm = Math.sqrt(nextU.reduce((s, x) => s + x*x, 0));
                u = norm > 0 ? nextU.map(x => x / norm) : u;
            }
            return u;
        };

        const u = powerIteration(centered);
        // Orthogonalize second axis
        let v = Array.from({length: vocab.length}, () => Math.random() - 0.5);
        const uProjection = dotProduct(v, u);
        v = v.map((x, i) => x - uProjection * u[i]); // subtract component along u
        v = powerIteration(centered.map(doc => {
            const proj = dotProduct(doc, u);
            return doc.map((x, i) => x - proj * u[i]);
        }));

        // Project centered coordinates
        const coords = centered.map(doc => ({
            x: dotProduct(doc, u),
            y: dotProduct(doc, v)
        }));

        // Map coordinates onto SVG plot coordinates
        renderPlot(coords, sentences, query, sortedResults);
    }

    function renderPlot(coords, sentences, query, sortedResults) {
        vectorSvg.innerHTML = '';
        const width = vectorSvg.clientWidth || 400;
        const height = vectorSvg.clientHeight || 400;
        const cx = width / 2;
        const cy = height / 2;

        // Draw axes lines
        const drawLine = (x1, y1, x2, y2, color, isDashed = false) => {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x1);
            line.setAttribute('y1', y1);
            line.setAttribute('x2', x2);
            line.setAttribute('y2', y2);
            line.setAttribute('stroke', color);
            line.setAttribute('stroke-width', 1);
            if (isDashed) line.setAttribute('stroke-dasharray', '4');
            vectorSvg.appendChild(line);
        };

        drawLine(20, cy, width - 20, cy, '#1e293b');
        drawLine(cx, 20, cx, height - 20, '#1e293b');

        // Scale factors
        let maxVal = 0.01;
        coords.forEach(c => {
            maxVal = Math.max(maxVal, Math.abs(c.x), Math.abs(c.y));
        });
        const scale = (Math.min(width, height) / 2 - 50) / maxVal;

        // Plot Sentence Points
        for (let i = 0; i < coords.length - 1; i++) {
            const px = cx + coords[i].x * scale;
            const py = cy - coords[i].y * scale; // Invert SVG y coordinate
            
            // Highlight text node color depending on cosine similarity score
            const score = sortedResults.find(r => r.text === sentences[i]).score;
            const nodeColor = score > 0.3 ? 'rgb(6, 182, 212)' : 'rgba(148, 163, 184, 0.6)';

            // Draw line to origin
            drawLine(cx, cy, px, py, 'rgba(30, 41, 59, 0.4)', true);

            // Draw dot
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', px);
            circle.setAttribute('cy', py);
            circle.setAttribute('r', 6);
            circle.setAttribute('fill', nodeColor);
            circle.setAttribute('stroke', '#ffffff');
            circle.setAttribute('stroke-width', 1.5);
            vectorSvg.appendChild(circle);

            // Add text label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', px + 10);
            text.setAttribute('y', py + 4);
            text.setAttribute('fill', '#94a3b8');
            text.setAttribute('font-size', '10px');
            text.setAttribute('font-weight', '500');
            text.textContent = `S${i + 1}`;
            vectorSvg.appendChild(text);
        }

        // Plot Query Vector (Neon Cyan Arrow)
        const qIdx = coords.length - 1;
        const qx = cx + coords[qIdx].x * scale;
        const qy = cy - coords[qIdx].y * scale;

        drawLine(cx, cy, qx, qy, 'rgb(234, 88, 12)', false); // Solid orange line

        const queryCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        queryCircle.setAttribute('cx', qx);
        queryCircle.setAttribute('cy', qy);
        queryCircle.setAttribute('r', 8);
        queryCircle.setAttribute('fill', 'rgb(234, 88, 12)');
        queryCircle.setAttribute('stroke', '#ffffff');
        queryCircle.setAttribute('stroke-width', 2);
        vectorSvg.appendChild(queryCircle);

        const queryText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        queryText.setAttribute('x', qx + 12);
        queryText.setAttribute('y', qy + 4);
        queryText.setAttribute('fill', 'white');
        queryText.setAttribute('font-size', '11px');
        queryText.setAttribute('font-weight', '700');
        queryText.textContent = `Query: "${query}"`;
        vectorSvg.appendChild(queryText);
    }

    // Attach search trigger
    searchBtn.addEventListener('click', executeSearch);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') executeSearch();
    });

    // Execute first search representation
    executeSearch();
}


/* ==========================================
   TAB 4: TOKENIZER LAB
   ========================================== */
function initTokenizerLab() {
    const textInput = document.getElementById('tokenizer-input-text');
    const chipContainer = document.getElementById('tokenizer-output-chips');
    const tokenCountLbl = document.getElementById('token-count-lbl');
    const charCountLbl = document.getElementById('char-count-lbl');

    const tempSlider = document.getElementById('llm-temp-slider');
    const tempVal = document.getElementById('llm-temp-val');
    const toppSlider = document.getElementById('llm-topp-slider');
    const toppVal = document.getElementById('llm-topp-val');
    
    const probChart = document.getElementById('probability-chart');
    const generateBtn = document.getElementById('llm-generate-btn');
    const resetBtn = document.getElementById('llm-reset-btn');
    const generatedBox = document.getElementById('llm-generated-output');

    // Vocabulary for mock BPE Tokenization
    const vocabList = [
        "Gen", "erative", "art", "ificial", "intell", "igence", "deep", 
        "neur", "al", "net", "work", "trans", "former", "learn", 
        "ing", "data", "scie", "nce", "model", "train", "techno", 
        "logy", "algor", "ithm", "comput", "e", "s", "ing", "ed"
    ];

    function runTokenizer() {
        const text = textInput.value;
        charCountLbl.textContent = text.length;

        if (text.trim() === '') {
            chipContainer.innerHTML = '';
            tokenCountLbl.textContent = '0';
            return;
        }

        // Mock BPE Tokenization algorithm (greedy prefix match)
        const words = text.split(/(\s+)/);
        let tokens = [];

        words.forEach(word => {
            if (word.trim() === '') {
                tokens.push({ text: word, id: 32 }); // White space token
                return;
            }

            let rem = word;
            while (rem.length > 0) {
                let matched = false;
                // Try to match vocabulary prefixes
                for (let i = 0; i < vocabList.length; i++) {
                    const prefix = vocabList[i];
                    if (rem.toLowerCase().startsWith(prefix.toLowerCase())) {
                        tokens.push({ text: rem.substring(0, prefix.length), id: i });
                        rem = rem.substring(prefix.length);
                        matched = true;
                        break;
                    }
                }
                // Fallback to character tokenization
                if (!matched) {
                    tokens.push({ text: rem[0], id: 100 + rem.charCodeAt(0) % 50 });
                    rem = rem.substring(1);
                }
            }
        });

        // Render color token chips
        chipContainer.innerHTML = '';
        tokens.forEach((tok, index) => {
            if (tok.text.trim() === '') {
                const spaceSpan = document.createElement('span');
                spaceSpan.innerHTML = '&nbsp;';
                chipContainer.appendChild(spaceSpan);
            } else {
                const chip = document.createElement('span');
                chip.className = `token-word token-chip-${tok.id % 6}`;
                chip.textContent = tok.text;
                chip.title = `Token ID: ${tok.id}`;
                chipContainer.appendChild(chip);
            }
        });

        tokenCountLbl.textContent = tokens.length;
    }

    // Next Token Generation Tree Simulator
    const generationTrees = {
        "is": [
            { w: "training", p: 0.40 },
            { w: "fine-tuning", p: 0.25 },
            { w: "prompting", p: 0.15 },
            { w: "learning", p: 0.10 },
            { w: "coding", p: 0.10 }
        ],
        "training": [
            { w: "deep", p: 0.45 },
            { w: "neural", p: 0.35 },
            { w: "supervised", p: 0.12 },
            { w: "robust", p: 0.08 }
        ],
        "neural": [
            { w: "networks", p: 0.75 },
            { w: "models", p: 0.15 },
            { w: "weights", p: 0.10 }
        ],
        "networks": [
            { w: "accurately", p: 0.40 },
            { w: "efficiently", p: 0.30 },
            { w: "first", p: 0.20 },
            { w: "locally", p: 0.10 }
        ],
        "models": [
            { w: "with", p: 0.50 },
            { w: "locally", p: 0.30 },
            { w: "using", p: 0.20 }
        ],
        "deep": [
            { w: "learning", p: 0.85 },
            { w: "transformers", p: 0.15 }
        ],
        "learning": [
            { w: "frameworks", p: 0.50 },
            { w: "methods", p: 0.30 },
            { w: "paradigms", p: 0.20 }
        ]
    };

    const defaultPredictions = [
        { w: "algorithms", p: 0.35 },
        { w: "data", p: 0.25 },
        { w: "insights", p: 0.20 },
        { w: "structures", p: 0.12 },
        { w: "code", p: 0.08 }
    ];

    let currentPromptText = "The best way to build AI models is";

    function updateProbabilityBars() {
        const words = currentPromptText.trim().split(/\s+/);
        const lastWord = words[words.length - 1].toLowerCase();
        
        // Retrieve base predictions
        let candidates = generationTrees[lastWord] || defaultPredictions;
        
        // 1. Apply Temperature Scaling
        const temp = parseFloat(tempSlider.value);
        let scaledSum = 0;
        
        const scaledProbabilities = candidates.map(c => {
            // Softmax scale logic: p' = exp(log(p) / T)
            const logP = Math.log(c.p);
            const scaledVal = Math.exp(logP / temp);
            scaledSum += scaledVal;
            return { w: c.w, val: scaledVal };
        });
        
        // Normalize probabilities
        scaledProbabilities.forEach(sp => sp.p = sp.val / scaledSum);

        // 2. Apply Top-P (Nucleus Sampling) filtering
        const topP = parseFloat(toppSlider.value);
        // Sort descending
        scaledProbabilities.sort((a, b) => b.p - a.p);
        
        let cumulativeP = 0;
        const filteredProbabilities = [];
        
        for (let i = 0; i < scaledProbabilities.length; i++) {
            const item = scaledProbabilities[i];
            cumulativeP += item.p;
            filteredProbabilities.push(item);
            if (cumulativeP >= topP) break; // Drop remaining low probability choices
        }

        // Render chart bars
        probChart.innerHTML = '';
        filteredProbabilities.forEach(item => {
            const pct = (item.p * 100).toFixed(1);
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';
            probItem.innerHTML = `
                <div class="prob-item-header">
                    <span class="prob-word">${item.w}</span>
                    <span class="prob-percentage">${pct}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: ${pct}%"></div>
                </div>
            `;
            probChart.appendChild(probItem);
        });

        return filteredProbabilities;
    }

    function generateNextToken() {
        const probs = updateProbabilityBars();
        if (probs.length === 0) return;
        
        // Sample next word based on probability distribution
        const r = Math.random();
        let cumulative = 0;
        let selectedWord = probs[probs.length - 1].w; // Fallback
        
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i].p;
            if (r <= cumulative) {
                selectedWord = probs[i].w;
                break;
            }
        }

        // Add to prompt context
        currentPromptText += " " + selectedWord;
        generatedBox.textContent = currentPromptText + "...";

        // Re-calculate distribution
        updateProbabilityBars();
    }

    // Bind Tokenizer Events
    textInput.addEventListener('input', runTokenizer);
    runTokenizer(); // Run on startup

    // Bind LLM Params Events
    tempSlider.addEventListener('input', (e) => {
        tempVal.textContent = e.target.value;
        updateProbabilityBars();
    });
    toppSlider.addEventListener('input', (e) => {
        toppVal.textContent = e.target.value;
        updateProbabilityBars();
    });

    generateBtn.addEventListener('click', generateNextToken);
    
    resetBtn.addEventListener('click', () => {
        currentPromptText = "The best way to build AI models is";
        generatedBox.textContent = currentPromptText + "...";
        updateProbabilityBars();
    });

    // Run first generation state
    updateProbabilityBars();
}


/* ==========================================
   TAB 5: DATA SCIENCE CLUB PORTAL
   ========================================== */
function initClubPortal() {
    const contactForm = document.getElementById('club-contact-form');
    
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const name = document.getElementById('contact-name').value;
        const email = document.getElementById('contact-email').value;
        
        showToast(`Thank you, ${name}! Your application has been logged.`);
        contactForm.reset();
    });
}


/* ==========================================
   TAB 6: QUIZ ARENA
   ========================================== */
function initQuizArena() {
    // UI Elements
    const startView = document.getElementById('quiz-start-view');
    const gameView = document.getElementById('quiz-game-view');
    const completeView = document.getElementById('quiz-complete-view');
    
    const startBtn = document.getElementById('quiz-start-btn');
    const nextBtn = document.getElementById('quiz-next-btn');
    const retryBtn = document.getElementById('quiz-retry-btn');
    
    const questionNumLbl = document.getElementById('quiz-question-number-lbl');
    const questionTxt = document.getElementById('quiz-question-txt');
    const optionsBox = document.getElementById('quiz-options-box');
    const explanationBox = document.getElementById('quiz-explanation-box');
    const progressFill = document.getElementById('quiz-progress-fill');
    
    const finalScoreLbl = document.getElementById('quiz-final-score');
    const nameInput = document.getElementById('quiz-name-input');
    const submitScoreBtn = document.getElementById('quiz-submit-score-btn');
    const leaderboardRows = document.getElementById('leaderboard-rows');

    // Quiz Questions Data Bank
    const quizQuestions = [
        {
            q: "Which activation function outputs values in the range of [-1, 1]?",
            options: ["ReLU", "Tanh", "Sigmoid", "Softmax"],
            ans: 1,
            exp: "Tanh (Hyperbolic Tangent) maps any real-valued number to the range [-1, 1], unlike Sigmoid which maps to [0, 1]."
        },
        {
            q: "What does the 'Temperature' parameter scale in Large Language Models?",
            options: ["Learning rate", "L2 weight penalty", "Softmax logits", "Transformer dimension"],
            ans: 2,
            exp: "Temperature scales the logits before the Softmax function is applied. Lowering it makes predictions more focused; raising it spreads probabilities."
        },
        {
            q: "In Retrieval-Augmented Generation (RAG), which formula is commonly used to find context similarity?",
            options: ["Binary Cross Entropy", "Euclidean Distance", "Mean Squared Error", "Cosine Similarity"],
            ans: 3,
            exp: "Cosine Similarity evaluates the angle between document embeddings, measuring semantic relevance regardless of snippet length."
        },
        {
            q: "Why is Backpropagation used during Neural Network training?",
            options: ["To compute layer outputs", "To calculate loss gradients", "To scale dataset features", "To initialize weights"],
            ans: 1,
            exp: "Backpropagation applies the calculus chain rule backward from the output layer to compute gradients of the loss function with respect to weights."
        },
        {
            q: "What is the primary role of a 'Tokenizer' in Generative AI architectures?",
            options: ["To translate texts to other languages", "To convert characters/words into numeric tokens", "To compile neural layers", "To reduce weights matrix size"],
            ans: 1,
            exp: "Tokenizers break raw strings into words or sub-words, representing them as unique numeric token IDs that the LLM embedding layer can process."
        }
    ];

    // Quiz State Variables
    let currentIdx = 0;
    let score = 0;
    let answered = false;

    // Leaderboard state
    let leaderboard = [];

    function loadLeaderboard() {
        const stored = localStorage.getItem('nexus_leaderboard');
        if (stored) {
            leaderboard = JSON.parse(stored);
        } else {
            // Seed mock high scores
            leaderboard = [
                { name: "Ben Carter", score: 500 },
                { name: "Priya Sharma", score: 400 },
                { name: "Leo Rodriguez", score: 300 }
            ];
            saveLeaderboard();
        }
        renderLeaderboard();
    }

    function saveLeaderboard() {
        localStorage.setItem('nexus_leaderboard', JSON.stringify(leaderboard));
    }

    function renderLeaderboard() {
        leaderboardRows.innerHTML = '';
        // Sort score desc
        const sorted = [...leaderboard].sort((a,b) => b.score - a.score);
        
        sorted.forEach((entry, idx) => {
            let rankClass = '';
            let rankText = `${idx + 1}th`;
            if (idx === 0) { rankClass = 'rank-gold'; rankText = '1st'; }
            else if (idx === 1) { rankClass = 'rank-silver'; rankText = '2nd'; }
            else if (idx === 2) { rankClass = 'rank-bronze'; rankText = '3rd'; }

            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="leaderboard-rank ${rankClass}">${rankText}</td>
                <td>${entry.name}</td>
                <td class="leaderboard-score" style="text-align:right;">${entry.score}</td>
            `;
            leaderboardRows.appendChild(row);
        });
    }

    function startQuiz() {
        currentIdx = 0;
        score = 0;
        answered = false;
        
        startView.style.display = 'none';
        completeView.style.display = 'none';
        gameView.style.display = 'block';
        
        showQuestion();
    }

    function showQuestion() {
        answered = false;
        nextBtn.style.display = 'none';
        explanationBox.style.display = 'none';
        explanationBox.className = 'quiz-feedback-box';
        
        const question = quizQuestions[currentIdx];
        
        // Update labels
        questionNumLbl.textContent = `Question ${currentIdx + 1} of 5`;
        questionTxt.textContent = question.q;
        progressFill.style.width = `${(currentIdx / quizQuestions.length) * 100}%`;

        // Render option buttons
        optionsBox.innerHTML = '';
        question.options.forEach((opt, idx) => {
            const btn = document.createElement('button');
            btn.className = 'quiz-option-btn';
            btn.textContent = opt;
            btn.addEventListener('click', () => selectOption(idx, btn));
            optionsBox.appendChild(btn);
        });
    }

    function selectOption(idx, clickedBtn) {
        if (answered) return;
        answered = true;

        const question = quizQuestions[currentIdx];
        const buttons = optionsBox.querySelectorAll('.quiz-option-btn');

        if (idx === question.ans) {
            // Correct choice
            clickedBtn.classList.add('correct');
            score += 100;
            explanationBox.classList.add('correct');
            explanationBox.innerHTML = `<strong>Correct!</strong> ${question.exp}`;
        } else {
            // Incorrect choice
            clickedBtn.classList.add('incorrect');
            buttons[question.ans].classList.add('correct'); // Highlight answer
            explanationBox.classList.add('incorrect');
            explanationBox.innerHTML = `<strong>Incorrect.</strong> ${question.exp}`;
        }

        explanationBox.style.display = 'block';
        nextBtn.style.display = 'inline-flex';
    }

    function handleNext() {
        currentIdx++;
        if (currentIdx < quizQuestions.length) {
            showQuestion();
        } else {
            // End of Quiz
            progressFill.style.width = '100%';
            gameView.style.display = 'none';
            completeView.style.display = 'block';
            finalScoreLbl.textContent = score;
        }
    }

    function submitScore() {
        const name = nameInput.value.trim();
        if (name === '') {
            showToast("Please enter your name.");
            return;
        }
        
        // Add score to board
        leaderboard.push({ name: name, score: score });
        saveLeaderboard();
        renderLeaderboard();
        
        showToast("Score submitted successfully!");
        nameInput.value = '';
        submitScoreBtn.disabled = true;
    }

    // Attach Event Listeners
    startBtn.addEventListener('click', startQuiz);
    nextBtn.addEventListener('click', handleNext);
    retryBtn.addEventListener('click', () => {
        submitScoreBtn.disabled = false;
        startQuiz();
    });
    submitScoreBtn.addEventListener('click', submitScore);

    // Initial Load
    loadLeaderboard();
}


/* ==========================================
   TAB 7: RIYA CHAT & RAG SIMULATOR
   ========================================== */
function initAIChatConsole() {
    const layoutWrapper = document.getElementById('chat-layout-wrapper');
    const ragToggleBtn = document.getElementById('chat-rag-toggle-btn');
    const ragToggleIcon = document.getElementById('chat-rag-toggle-icon');
    const messagesBox = document.getElementById('chat-messages-box');
    const inputField = document.getElementById('chat-input-field');
    const sendBtn = document.getElementById('chat-send-btn');

    // RAG Pipeline Panel Elements
    const ragVectorLog = document.getElementById('rag-query-vector');
    const ragMatchesBox = document.getElementById('rag-vector-matches');
    const ragPromptLog = document.getElementById('rag-augmented-prompt');

    let isRAGMode = false;

    // Local Vector Database documents
    const docCorpus = [
        {
            title: "Supervised Learning",
            text: "Supervised learning is a machine learning paradigm where models are trained on labeled data. The model learns a mapping function from input variables to output variables by minimizing a loss function (e.g. Mean Squared Error)."
        },
        {
            title: "Neural Networks",
            text: "Neural networks are computational models inspired by biological brains. They consist of connected nodes or neurons organized in layers: input, hidden, and output layers. Nodes adjust connection weights using backpropagation algorithms."
        },
        {
            title: "Large Language Models",
            text: "Large Language Models (LLMs) are deep learning models trained on vast text corpora to predict the next token. They use self-attention mechanisms in Transformer architectures to capture long-range text dependencies."
        },
        {
            title: "Retrieval-Augmented Generation",
            text: "Retrieval-Augmented Generation (RAG) is a technique that retrieves relevant document snippets from a vector database and appends them to the LLM prompt. This reduces model hallucinations and incorporates external knowledge."
        },
        {
            title: "Vector Embeddings",
            text: "Vector embeddings represent text as high-dimensional numerical coordinates. Text pieces with similar semantic meanings are placed closer together in this vector space, allowing for fast cosine similarity search."
        }
    ];

    // Toggle RAG Mode split screen
    ragToggleBtn.addEventListener('click', () => {
        isRAGMode = !isRAGMode;
        if (isRAGMode) {
            layoutWrapper.classList.add('split-view');
            ragToggleIcon.setAttribute('data-lucide', 'toggle-right');
            ragToggleIcon.style.color = 'var(--accent-cyan)';
            showToast("RAG Pipeline Simulator Activated.");
        } else {
            layoutWrapper.classList.remove('split-view');
            ragToggleIcon.setAttribute('data-lucide', 'toggle-left');
            ragToggleIcon.style.color = 'var(--text-muted)';
            showToast("Standard Chat Mode.");
        }
        lucide.createIcons();
    });

    // Send chat message
    function sendMessage() {
        const text = inputField.value.trim();
        if (text === '') return;

        // Render User Bubble
        renderMessageBubble(text, 'user', 'You');
        inputField.value = '';

        // Render Typing Indicator
        const typingId = renderTypingIndicator();

        if (isRAGMode) {
            // Process Retrieval-Augmented Generation Flow
            setTimeout(() => {
                processRAGFlow(text, typingId);
            }, 800);
        } else {
            // Process Standard Chat reply
            setTimeout(() => {
                const response = getStaticResponse(text);
                removeTypingIndicator(typingId);
                renderMessageBubble(response, 'bot', 'Riya AI');
            }, 1000);
        }
    }

    function renderMessageBubble(text, senderClass, senderName) {
        const bubble = document.createElement('div');
        bubble.className = `chat-msg-bubble ${senderClass}`;
        bubble.innerHTML = `
            <span class="chat-msg-sender">${senderName}</span>
            <div class="chat-msg-body">${text}</div>
        `;
        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function renderTypingIndicator() {
        const id = 'typing-' + Date.now();
        const bubble = document.createElement('div');
        bubble.className = 'chat-msg-bubble bot';
        bubble.id = id;
        bubble.innerHTML = `
            <span class="chat-msg-sender">Riya AI</span>
            <div class="chat-msg-body">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        messagesBox.appendChild(bubble);
        messagesBox.scrollTop = messagesBox.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        const indicator = document.getElementById(id);
        if (indicator) indicator.remove();
    }

    // Standard static responses
    function getStaticResponse(query) {
        const q = query.toLowerCase();
        if (q.includes('hello') || q.includes('hi')) {
            return "Hello! How can I assist you with your machine learning journey today?";
        } else if (q.includes('neural network')) {
            return "Neural networks are models inspired by human brains. They stack layers of parameters (weights/biases) that adapt via optimization algorithms like backpropagation.";
        } else if (q.includes('embed') || q.includes('vector')) {
            return "Embeddings translate vocabulary words into continuous coordinates in vector space. Words with close meanings exhibit smaller angular distances.";
        } else if (q.includes('rag')) {
            return "RAG stands for Retrieval-Augmented Generation. Try activating RAG Mode using the toggle in the header to observe it pull documents in real-time!";
        } else {
            return "Interesting question! Machine Learning models learn patterns from training distributions. What specific component (like loss functions, classifiers, or prompting parameters) would you like to explore?";
        }
    }

    // Simulated RAG Pipeline Math and Display
    function processRAGFlow(query, typingId) {
        // 1. Vocabulary Extraction
        const getWords = (txt) => txt.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 2);
        const queryWords = getWords(query);

        // Vector representing query
        const queryVecArr = queryWords.map(() => (Math.random() * 0.9).toFixed(3));
        ragVectorLog.textContent = `[${queryVecArr.slice(0, 4).join(', ')}, ...]`;

        // 2. Score Document Similarity
        // Calculate a mock overlap score: number of query words appearing in documents + random float
        const matches = docCorpus.map(doc => {
            const docWords = getWords(doc.text);
            let matchCount = 0;
            queryWords.forEach(w => {
                if (docWords.includes(w)) matchCount++;
            });
            const score = matchCount > 0 ? (matchCount / queryWords.length) * 0.7 + 0.3 * Math.random() : 0.1 * Math.random();
            return { title: doc.title, text: doc.text, score: score };
        });

        // Sort descending
        matches.sort((a, b) => b.score - a.score);

        // 3. Render Match debug results
        ragMatchesBox.innerHTML = '';
        matches.forEach((m, i) => {
            const activeClass = i < 2 ? 'active' : ''; // Top 2 are retrieved
            const badge = document.createElement('div');
            badge.className = `rag-match-chip ${activeClass}`;
            badge.innerHTML = `
                <div style="display:flex; justify-content:between; font-weight:600; font-size:11px; margin-bottom:4px;">
                    <span>${m.title}</span>
                    <span style="color:var(--accent-cyan)">Sim: ${m.score.toFixed(3)}</span>
                </div>
                <div>"${m.text.substring(0, 80)}..."</div>
            `;
            ragMatchesBox.appendChild(badge);
        });

        // 4. Augment Prompt sent to model
        const retrievedChunks = `[Retrieved Chunk 1: ${matches[0].title}]\n"${matches[0].text}"\n\n[Retrieved Chunk 2: ${matches[1].title}]\n"${matches[1].text}"`;
        
        const finalPrompt = `System: Answer the question using the context provided.\n\nContext:\n${retrievedChunks}\n\nUser Question: ${query}`;
        ragPromptLog.textContent = finalPrompt;

        // 5. Stream Answer
        removeTypingIndicator(typingId);
        
        // Synthesize dynamic answer using context
        const responseText = `Based on my retrieved knowledge regarding **${matches[0].title}** and **${matches[1].title}**, here is what I found:\n\n${matches[0].text}\n\nAdditionally, ${matches[1].text.replace(/^\w/, c => c.toLowerCase())}`;
        
        renderMessageBubble(responseText, 'bot', 'Riya AI (RAG Mode)');
    }

    // Bind triggers
    sendBtn.addEventListener('click', sendMessage);
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}
