import React from 'react';
import Header from "../header/Header.jsx";
import "./CropPage.css";
import { TextField } from "@mui/material";
import { useNavigate } from "react-router-dom";
import LinearProgress from '@mui/material/LinearProgress';

//--------------------------------------------------------------------


class TreeNode {
    constructor(condition, trueNode, falseNode, label) {
        this.condition = condition;
        this.trueNode = trueNode;
        this.falseNode = falseNode;
        this.label = label;
    }
}

const decisionTree = new TreeNode(
    // First level condition
    (N, P, K, temperature, humidity, pH, rainfall) =>
        N >= 24.5 && N <= 57.2 &&
        P >= 49.5 && P <= 77.2 &&
        K >= 43.8 && K <= 67.2 &&
        temperature >= 22.7 && temperature <= 26.6 &&
        humidity >= 69.4 && humidity <= 78.2 &&
        pH >= 5.3 && pH <= 6.4 &&
        rainfall >= 70.3 && rainfall <= 82.9,

    // Second level nodes
    new TreeNode(
        // Condition for 'apple'
        null, null, null, 'apple'
    ),
    new TreeNode(
        // Condition for 'banana'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 90.7 && N <= 108.8 &&
            P >= 75.5 && P <= 92.1 &&
            K >= 50.7 && K <= 55.8 &&
            temperature >= 26.1 && temperature <= 28.6 &&
            humidity >= 76.5 && humidity <= 80.1 &&
            pH >= 5.4 && pH <= 5.8 &&
            rainfall >= 92.1 && rainfall <= 98.1,

        // Third level nodes
        new TreeNode(
            // Condition for 'banana' - True branch
            null, null, null, 'banana'
        ),
        new TreeNode(
            // Condition for 'banana' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'blackgram'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 29.1 && N <= 55.2 &&
            P >= 66.8 && P <= 78.7 &&
            K >= 21.1 && K <= 30.8 &&
            temperature >= 25.3 && temperature <= 32.3 &&
            humidity >= 63.4 && humidity <= 67.9 &&
            pH >= 6.0 && pH <= 6.7 &&
            rainfall >= 62.6 && rainfall <= 68.2,

        // Third level nodes
        new TreeNode(
            // Condition for 'blackgram' - True branch
            null, null, null, 'blackgram'
        ),
        new TreeNode(
            // Condition for 'blackgram' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'chickpea'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 24.8 && N <= 51.3 &&
            P >= 65.4 && P <= 75.2 &&
            K >= 79.4 && K <= 82.1 &&
            temperature >= 17.9 && temperature <= 19.9 &&
            humidity >= 15.4 && humidity <= 16.8 &&
            pH >= 5.0 && pH <= 7.3 &&
            rainfall >= 69.5 && rainfall <= 76.8,

        // Third level nodes
        new TreeNode(
            // Condition for 'chickpea' - True branch
            null, null, null, 'chickpea'
        ),
        new TreeNode(
            // Condition for 'chickpea' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'coconut'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 19.6 && N <= 33.1 &&
            P >= 7.2 && P <= 27.9 &&
            K >= 26.8 && K <= 29.6 &&
            temperature >= 25.2 && temperature <= 27.4 &&
            humidity >= 92.4 && humidity <= 99.2 &&
            pH >= 5.0 && pH <= 5.4 &&
            rainfall >= 100.0 && rainfall <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'coconut' - True branch
            null, null, null, 'coconut'
        ),
        new TreeNode(
            // Condition for 'coconut' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'coffee'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 84.5 && N <= 118.3 &&
            P >= 16.8 && P <= 32.2 &&
            K >= 28.6 && K <= 33.3 &&
            temperature >= 23.4 && temperature <= 25.4 &&
            humidity >= 53.2 && humidity <= 59.4 &&
            pH >= 6.3 && pH <= 6.8 &&
            rainfall >= 100.0 && rainfall <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'coffee' - True branch
            null, null, null, 'coffee'
        ),
        new TreeNode(
            // Condition for 'coffee' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'cotton'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 105.7 && N <= 134.3 &&
            P >= 37.5 && P <= 54.1 &&
            K >= 17.8 && K <= 21.8 &&
            temperature >= 22.8 && temperature <= 24.0 &&
            humidity >= 79.2 && humidity <= 80.9 &&
            pH >= 5.2 && pH <= 6.5 &&
            rainfall >= 65.3 && rainfall <= 87.4,

        // Third level nodes
        new TreeNode(
            // Condition for 'cotton' - True branch
            null, null, null, 'cotton'
        ),
        new TreeNode(
            // Condition for 'cotton' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'grapes'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 13.2 && N <= 39.2 &&
            P >= 134.4 && P <= 139.2 &&
            K >= 202.0 && K <= 203.2 &&
            temperature >= 8.3 && temperature <= 41.0 &&
            humidity >= 80.7 && humidity <= 82.3 &&
            pH >= 5.4 && pH <= 5.7 &&
            rainfall >= 67.6 && rainfall <= 73.6,

        // Third level nodes
        new TreeNode(
            // Condition for 'grapes' - True branch
            null, null, null, 'grapes'
        ),
        new TreeNode(
            // Condition for 'grapes' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'jute'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 68.0 && N <= 78.8 &&
            P >= 40.5 && P <= 51.9 &&
            K >= 36.3 && K <= 42.4 &&
            temperature >= 23.2 && temperature <= 24.4 &&
            humidity >= 72.6 && humidity <= 87.0 &&
            pH >= 6.0 && pH <= 6.4 &&
            rainfall >= 100.0 && rainfall <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'jute' - True branch
            null, null, null, 'jute'
        ),
        new TreeNode(
            // Condition for 'jute' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'kidneybeans'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 18.1 && N <= 38.5 &&
            P >= 55.1 && P <= 77.3 &&
            K >= 15.1 && K <= 25.1 &&
            temperature >= 15.7 && temperature <= 23.3 &&
            humidity >= 19.8 && humidity <= 23.6 &&
            pH >= 5.4 && pH <= 5.5 &&
            rainfall >= 64.9 && rainfall <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'kidneybeans' - True branch
            null, null, null, 'kidneybeans'
        ),
        new TreeNode(
            // Condition for 'kidneybeans' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'lentil'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 15.7 && N <= 35.3 &&
            P >= 57.0 && P <= 78.1 &&
            K >= 16.7 && K <= 24.7 &&
            temperature >= 18.9 && temperature <= 28.0 &&
            humidity >= 62.8 && humidity <= 68.5 &&
            pH >= 5.1 && pH <= 6.2 &&
            rainfall >= 38.8 && rainfall <= 54.9,

        // Third level nodes
        new TreeNode(
            // Condition for 'lentil' - True branch
            null, null, null, 'lentil'
        ),
        new TreeNode(
            // Condition for 'lentil' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'maize'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 64.1 && N <= 97.2 &&
            P >= 38.3 && P <= 59.5 &&
            K >= 18.8 && K <= 22.2 &&
            temperature >= 19.5 && temperature <= 25.4 &&
            humidity >= 56.7 && humidity <= 71.5 &&
            pH >= 5.2 && pH <= 6.4 &&
            rainfall >= 61.2 && rainfall <= 92.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'maize' - True branch
            null, null, null, 'maize'
        ),
        new TreeNode(
            // Condition for 'maize' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'mango'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 15.2 && N <= 38.2 &&
            P >= 17.6 && P <= 34.2 &&
            K >= 28.1 && K <= 34.5 &&
            temperature >= 27.5 && temperature <= 33.5 &&
            humidity >= 46.8 && humidity <= 53.2 &&
            pH >= 4.5 && pH <= 5.8 &&
            rainfall >= 84.6 && rainfall <= 96.9,

        // Third level nodes
        new TreeNode(
            // Condition for 'mango' - True branch
            null, null, null, 'mango'
        ),
        new TreeNode(
            // Condition for 'mango' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'mothbeans'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 12.1 && N <= 39.5 &&
            P >= 37.8 && P <= 57.5 &&
            K >= 17.0 && K <= 25.1 &&
            temperature >= 24.7 && temperature <= 31.3 &&
            humidity >= 42.2 && humidity <= 63.6 &&
            pH >= 3.8 && pH <= 9.0 &&
            rainfall >= 30.2 && rainfall <= 69.4,

        // Third level nodes
        new TreeNode(
            // Condition for 'mothbeans' - True branch
            null, null, null, 'mothbeans'
        ),
        new TreeNode(
            // Condition for 'mothbeans' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'mungbean'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 14.2 && N <= 39.6 &&
            P >= 37.9 && P <= 56.2 &&
            K >= 15.8 && K <= 25.4 &&
            temperature >= 27.3 && temperature <= 29.4 &&
            humidity >= 80.5 && humidity <= 88.9 &&
            pH >= 5.9 && pH <= 6.6 &&
            rainfall >= 37.5 && rainfall <= 58.6,

        // Third level nodes
        new TreeNode(
            // Condition for 'mungbean' - True branch
            null, null, null, 'mungbean'
        ),
        new TreeNode(
            // Condition for 'mungbean' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'muskmelon'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 83.3 && N <= 116.6 &&
            P >= 5.8 && P <= 30.4 &&
            K >= 48.3 && K <= 54.7 &&
            temperature >= 27.2 && temperature <= 28.7 &&
            humidity >= 91.4 && humidity <= 92.8 &&
            pH >= 6.0 && pH <= 6.2 &&
            rainfall >= 20.6 && rainfall <= 28.7,

        // Third level nodes
        new TreeNode(
            // Condition for 'muskmelon' - True branch
            null, null, null, 'muskmelon'
        ),
        new TreeNode(
            // Condition for 'muskmelon' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'orange'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 17.2 && N <= 37.8 &&
            P >= 6.7 && P <= 29.1 &&
            K >= 6.1 && K <= 13.1 &&
            temperature >= 11.2 && temperature <= 33.1 &&
            humidity >= 91.7 && humidity <= 93.2 &&
            pH >= 6.1 && pH <= 6.7 &&
            rainfall >= 100.0 && rainfall <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'orange' - True branch
            null, null, null, 'orange'
        ),
        new TreeNode(
            // Condition for 'orange' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'papaya'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 33.0 && N <= 57.5 &&
            P >= 46.4 && P <= 69.3 &&
            K >= 46.0 && K <= 55.4 &&
            temperature >= 23.5 && temperature <= 42.4 &&
            humidity >= 91.6 && humidity <= 93.3 &&
            pH >= 6.0 && pH <= 6.2 &&
            rainfall >= 39.9 && rainfall <= 98.6,

        // Third level nodes
        new TreeNode(
            // Condition for 'papaya' - True branch
            null, null, null, 'papaya'
        ),
        new TreeNode(
            // Condition for 'papaya' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'pigeonpeas'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 19.2 && N <= 39.3 &&
            P >= 55.4 && P <= 79.3 &&
            K >= 16.8 && K <= 24.9 &&
            temperature >= 19.1 && temperature <= 36.2 &&
            humidity >= 30.9 && humidity <= 68.3 &&
            pH >= 4.4 && pH <= 7.3 &&
            rainfall >= 92.7 && rainfall <= 98.5,

        // Third level nodes
        new TreeNode(
            // Condition for 'pigeonpeas' - True branch
            null, null, null, 'pigeonpeas'
        ),
        new TreeNode(
            // Condition for 'pigeonpeas' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'pomegranate'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 20.1 && N <= 37.2 &&
            P >= 6.2 && P <= 29.2 &&
            K >= 36.6 && K <= 44.0 &&
            temperature >= 18.7 && temperature <= 24.0 &&
            humidity >= 86.2 && humidity <= 93.7 &&
            pH >= 5.3 && pH <= 7.0 &&
            rainfall >= 100.0 && N <= 100.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'pomegranate' - True branch
            null, null, null, 'pomegranate'
        ),
        new TreeNode(
            // Condition for 'pomegranate' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'rice'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 78.5 && N <= 100.0 &&
            P >= 13.6 && P <= 24.0 &&
            K >= 16.0 && K <= 30.2 &&
            temperature >= 16.6 && temperature <= 35.0 &&
            humidity >= 76.5 && humidity <= 89.6 &&
            pH >= 5.0 && pH <= 6.5 &&
            rainfall >= 145.5 && rainfall <= 160.0,

        // Third level nodes
        new TreeNode(
            // Condition for 'rice' - True branch
            null, null, null, 'rice'
        ),
        new TreeNode(
            // Condition for 'rice' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'watermelon'
        (N, P, K, temperature, humidity, pH, rainfall) =>
            N >= 76.6 && N <= 108.0 &&
            P >= 2.0 && P <= 18.8 &&
            K >= 46.1 && K <= 56.4 &&
            temperature >= 29.1 && temperature <= 30.8 &&
            humidity >= 80.0 && humidity <= 82.2 &&
            pH >= 6.1 && pH <= 6.7 &&
            rainfall >= 80.7 && rainfall <= 97.6,

        // Third level nodes
        new TreeNode(
            // Condition for 'watermelon' - True branch
            null, null, null, 'watermelon'
        ),
        new TreeNode(
            // Condition for 'watermelon' - False branch
            null, null, null, 'unknown'
        )
    ),
    new TreeNode(
        // Condition for 'unknown'
        null, null, null, 'unknown'
    )
);

// Focus on Empty Input fields
function focusEmptyFields() {
    // list of all the input elements
    const inputElements = [
        document.getElementById('nitrogen-crop-input'),
        document.getElementById('temp-crop-input'),
        document.getElementById('phosphorous-crop-input'),
        document.getElementById('humidity-crop-input'),
        document.getElementById('potassium-crop-input'),
        document.getElementById('ph-crop-input'),
        document.getElementById('rainfall-crop-input'),
    ];

    for (let i = 0; i < inputElements.length; i++) {
        if (inputElements[i].value === '') {
            inputElements[i].focus();
            return false;
        }
    }

    return true;
}

export const crop_value_ranges = {
    nitrogen: [0, 150], phosphorous: [5, 145], potassium: [5, 205], temperature: [0, 50],
    humidity: [1, 100], ph: [3, 10], rainfall: [20, 300]
};

function findSuitableCrop(node, N, P, K, temperature, humidity, pH, rainfall) {
    if (node.label !== null) {
        return node;
    }
    if (node.condition(N, P, K, temperature, humidity, pH, rainfall)) {
        return findSuitableCrop(node.trueNode, N, P, K, temperature, humidity, pH, rainfall);
    } else {
        return findSuitableCrop(node.falseNode, N, P, K, temperature, humidity, pH, rainfall);
    }
}

function handleClick(navigate) {
    const isFieldEmpty = focusEmptyFields();
    if (!isFieldEmpty) {
        console.log("Some Inputs are empty !");
        return;
    }

    // Get the values of all input fields
    const nitrogenValue = parseFloat(document.getElementById('nitrogen-crop-input').value);
    const tempValue = parseFloat(document.getElementById('temp-crop-input').value);
    const phosphorousValue = parseFloat(document.getElementById('phosphorous-crop-input').value);
    const humidityValue = parseFloat(document.getElementById('humidity-crop-input').value);
    const potassiumValue = parseFloat(document.getElementById('potassium-crop-input').value);
    const phValue = parseFloat(document.getElementById('ph-crop-input').value);
    const rainfallValue = parseFloat(document.getElementById('rainfall-crop-input').value);

    // Check if the Input values are in required ranges
    const min_temp = crop_value_ranges.temperature[0];
    const max_temp = crop_value_ranges.temperature[1];
    const min_humid = crop_value_ranges.humidity[0];
    const max_humid = crop_value_ranges.humidity[1];
    if (tempValue < min_temp || tempValue > max_temp) {
        window.alert("Temperature must be between 0-50 Celsius !");
        return;
    } else if (humidityValue < min_humid || humidityValue > max_humid) {
        window.alert("Humidity must be between 1-100 !");
        return;
    }

    // Make progressbar visible
    const progressBar = document.querySelector('.crop-progress-bar');
    progressBar.style.display = 'block';
    progressBar.style.visibility = 'visible';

    // Call the crop prediction function
    const predictedCropNode = findSuitableCrop(decisionTree, nitrogenValue, phosphorousValue, potassiumValue, tempValue, humidityValue, phValue, rainfallValue);

    // Redirect to Result page along with predicted crop
    navigate("/crop_result", { state: { predicted_crop: predictedCropNode.trueNode.label } });
}

//--------------------------------------------------------------------

export function CropPage() {
    const navigate = useNavigate();

    // Called when Enter is pressed
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            handleClick(navigate);
        }
    });

    return (
        <>
            <Header />
            <LinearProgress style={{ visibility: 'hidden', display: 'none' }} className="crop-progress-bar" color="success" />
            <p className="crop-p">
                Provide soil characteristics to identify the most suitable <b>CROP</b> for cultivation on your farm. 🌱🚜👨‍🌾
            </p>
            <div className="crop-container">
                <TextField id="nitrogen-crop-input" label="Ratio of Nitrogen" variant="outlined" color="success" type="number" />
                <TextField id="temp-crop-input" label="Temperature in Celsius" variant="outlined" color="success" type="number" inputProps={{ min: 5, max: 50 }} />
                <TextField id="phosphorous-crop-input" label="Ratio of Phosphorous" variant="outlined" color="success" type="number" />
                <TextField id="humidity-crop-input" label="% of Humidity" variant="outlined" color="success" type="number" />
                <TextField id="potassium-crop-input" label="Ratio of Potassium" variant="outlined" color="success" type="number" />
                <TextField id="ph-crop-input" label="PH Level of soil" variant="outlined" color="success" type="number" />
                <TextField id="rainfall-crop-input" label="Rainfall in Milimeter (mm)" variant="outlined" color="success" type="number" />
                <button className="predict_crop_btn" onClick={() => handleClick(navigate)}> PREDICT </button>
            </div>
        </>
    );
}