const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');

const app = express();

// Middleware
app.use(bodyParser.json());
app.use(cors());

// API endpoint to make predictions
app.post('/predict', (req, res) => {
    const inputData = req.body;

    // Options for PythonShell
    const options = {
        mode: 'text',
        pythonPath: 'python', // Use 'python3' if on Linux/Mac
        scriptPath: __dirname, // Path to the Python script
        args: [JSON.stringify(inputData)],
    };

    // Run the Python script
    PythonShell.run('predict.py', options, (err, results) => {
        if (err) {
            console.error(err);
            return res.status(500).json({ error: 'Prediction failed' });
        }

        try {
            // Parse the prediction result
            const predictionResult = JSON.parse(results[0]);
            const prediction = predictionResult.prediction[0]; // Extract the prediction value

            // Send the prediction as a response
            res.json({ prediction: prediction });
        } catch (parseError) {
            console.error('Error parsing prediction result:', parseError);
            res.status(500).json({ error: 'Failed to parse prediction result' });
        }
    });
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});