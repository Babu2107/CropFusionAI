import React, { useState } from 'react';
import Header from "../header/Header.jsx";
import "./CropPage.css";
import { TextField } from "@mui/material";
import { useNavigate } from "react-router-dom";
import LinearProgress from '@mui/material/LinearProgress';
import axios from 'axios';

const CROP_ENDPOINT = 'https://8080-797137136eb6451193a1f8c64a951490.onpatr.cloud/crop_recommend';

// Min-Max values of crop inputs
export const crop_value_ranges = {
  nitrogen: [0, 150],
  phosphorous: [5, 145],
  potassium: [5, 205],
  temperature: [0, 50],
  humidity: [1, 100],
  ph: [3, 10],
  rainfall: [20, 300]
};

async function findSuitableCrop(N, P, K, temperature, humidity, ph, rainfall) {
    const inputData = {
      N,
      P,
      K,
      temperature,
      humidity,
      ph,
      rainfall
    };
    console.log(inputData);
    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', inputData);

      return response.data.predicted_crop_name;
    } catch (error) {
      console.error('Error predicting crop:', error);
      return 'rice';
    }
}

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

  // Check if any of the input fields is empty & focus on it
  for (let i = 0; i < inputElements.length; i++) {
    if (inputElements[i].value === '') {
      inputElements[i].focus();
      return false;
    }
  }

  return true;
}

export function CropPage() {
  const navigate = useNavigate();
  const [predictedCropName, setPredictedCropName] = useState('');
  const [progressBarVisible, setProgressBarVisible] = useState(false);

  // Called when Button Clicked
  async function handleClick() {
    // Continue only if all fields are filled.
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
    setProgressBarVisible(true);

    // Call the crop prediction function
    const suitableCrop = await findSuitableCrop(
      nitrogenValue, phosphorousValue, potassiumValue,
      tempValue, humidityValue, phValue, rainfallValue
    );
    console.log(suitableCrop);
    window.alert(suitableCrop);

    // Redirect to Result page along with predicted crop
    navigate("/crop_result", { state: { predicted_crop: suitableCrop } });
  }

  // Called when Enter is pressed
  function handleKeyDown(event) {
    if (event.key === 'Enter') {
      handleClick();
    }
  }

  return (
    <>
      <Header />
      <LinearProgress style={{ visibility: progressBarVisible ? 'visible' : 'hidden', display: progressBarVisible ? 'block' : 'none' }} className="crop-progress-bar" color="success" />
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
        <button className="predict_crop_btn" onClick={handleClick}> PREDICT </button>
      </div>
    </>
  );
}