import Header from "../header/Header.jsx"
import "./FertilizerPage.css"
import { TextField } from "@mui/material"
import React, { useState } from 'react';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import { output_descriptions } from "./FertilizerOutputs.jsx";
import { useNavigate } from "react-router-dom";
import { crop_value_ranges } from "../crop/CropPage.jsx";
import LinearProgress from "@mui/material/LinearProgress";
import axios from "axios";

const cropData = [
    {
        label: 'apple',
        min_N: 24.5,
        max_N: 57.2,
        min_P: 49.5,
        max_P: 77.2,
        min_K: 43.8,
        max_K: 67.2,
        min_temperature: 22.7,
        max_temperature: 26.6,
        min_humidity: 69.4,
        max_humidity: 78.2,
        min_ph: 5.3,
        max_ph: 6.4,
        min_rainfall: 70.3,
        max_rainfall: 82.9,
    },
    {
        label: 'banana',
        min_N: 90.7,
        max_N: 108.8,
        min_P: 75.5,
        max_P: 92.1,
        min_K: 50.7,
        max_K: 55.8,
        min_temperature: 26.1,
        max_temperature: 28.6,
        min_humidity: 76.5,
        max_humidity: 80.1,
        min_ph: 5.4,
        max_ph: 5.8,
        min_rainfall: 92.1,
        max_rainfall: 98.1,
    },
    {
        label: 'blackgram',
        min_N: 29.1,
        max_N: 55.2,
        min_P: 66.8,
        max_P: 78.7,
        min_K: 21.1,
        max_K: 30.8,
        min_temperature: 25.3,
        max_temperature: 32.3,
        min_humidity: 63.4,
        max_humidity: 67.9,
        min_ph: 6.0,
        max_ph: 6.7,
        min_rainfall: 62.6,
        max_rainfall: 68.2,
    },
    {
        label: 'chickpea',
        min_N: 24.8,
        max_N: 51.3,
        min_P: 65.4,
        max_P: 75.2,
        min_K: 79.4,
        max_K: 82.1,
        min_temperature: 17.9,
        max_temperature: 19.9,
        min_humidity: 15.4,
        max_humidity: 16.8,
        min_ph: 5.0,
        max_ph: 7.3,
        min_rainfall: 69.5,
        max_rainfall: 76.8,
    },
    {
        label: 'coconut',
        min_N: 19.6,
        max_N: 33.1,
        min_P: 7.2,
        max_P: 27.9,
        min_K: 26.8,
        max_K: 29.6,
        min_temperature: 25.2,
        max_temperature: 27.4,
        min_humidity: 92.4,
        max_humidity: 99.2,
        min_ph: 5.0,
        max_ph: 5.4,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    {
        label: 'coffee',
        min_N: 84.5,
        max_N: 118.3,
        min_P: 16.8,
        max_P: 32.2,
        min_K: 28.6,
        max_K: 33.3,
        min_temperature: 23.4,
        max_temperature: 25.4,
        min_humidity: 53.2,
        max_humidity: 59.4,
        min_ph: 6.3,
        max_ph: 6.8,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    {
        label: 'cotton',
        min_N: 105.7,
        max_N: 134.3,
        min_P: 37.5,
        max_P: 54.1,
        min_K: 17.8,
        max_K: 21.8,
        min_temperature: 22.8,
        max_temperature: 24.0,
        min_humidity: 79.2,
        max_humidity: 80.9,
        min_ph: 5.2,
        max_ph: 6.5,
        min_rainfall: 65.3,
        max_rainfall: 87.4,
    },
    {
        label: 'grapes',
        min_N: 13.2,
        max_N: 39.2,
        min_P: 134.4,
        max_P: 139.2,
        min_K: 202.0,
        max_K: 203.2,
        min_temperature: 8.3,
        max_temperature: 41.0,
        min_humidity: 80.7,
        max_humidity: 82.3,
        min_ph: 5.4,
        max_ph: 5.7,
        min_rainfall: 67.6,
        max_rainfall: 73.6,
    },
    {
        label: 'jute',
        min_N: 68.0,
        max_N: 78.8,
        min_P: 40.5,
        max_P: 51.9,
        min_K: 36.3,
        max_K: 42.4,
        min_temperature: 23.2,
        max_temperature: 24.4,
        min_humidity: 72.6,
        max_humidity: 87.0,
        min_ph: 6.0,
        max_ph: 6.4,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    {
        label: 'kidneybeans',
        min_N: 18.1,
        max_N: 38.5,
        min_P: 55.1,
        max_P: 77.3,
        min_K: 15.1,
        max_K: 25.1,
        min_temperature: 15.7,
        max_temperature: 23.3,
        min_humidity: 19.8,
        max_humidity: 23.6,
        min_ph: 5.4,
        max_ph: 5.5,
        min_rainfall: 64.9,
        max_rainfall: 100.0,
    },
    {
        label: 'lentil',
        min_N: 15.7,
        max_N: 35.3,
        min_P: 57.0,
        max_P: 78.1,
        min_K: 16.7,
        max_K: 24.7,
        min_temperature: 18.9,
        max_temperature: 28.0,
        min_humidity: 62.8,
        max_humidity: 68.5,
        min_ph: 5.1,
        max_ph: 6.2,
        min_rainfall: 38.8,
        max_rainfall: 54.9,
    },
    {
        label: 'Maize',
        min_N: 64.1,
        max_N: 97.2,
        min_P: 38.3,
        max_P: 59.5,
        min_K: 18.8,
        max_K: 22.2,
        min_temperature: 19.5,
        max_temperature: 25.4,
        min_humidity: 56.7,
        max_humidity: 71.5,
        min_ph: 5.2,
        max_ph: 6.4,
        min_rainfall: 61.2,
        max_rainfall: 92.0,
    },
    {
        label: 'mango',
        min_N: 15.2,
        max_N: 38.2,
        min_P: 17.6,
        max_P: 34.2,
        min_K: 28.1,
        max_K: 34.5,
        min_temperature: 27.5,
        max_temperature: 33.5,
        min_humidity: 46.8,
        max_humidity: 53.2,
        min_ph: 4.5,
        max_ph: 5.8,
        min_rainfall: 84.6,
        max_rainfall: 96.9,
    },
    {
        label: 'mothbeans',
        min_N: 12.1,
        max_N: 39.5,
        min_P: 37.8,
        max_P: 57.5,
        min_K: 17.0,
        max_K: 25.1,
        min_temperature: 24.7,
        max_temperature: 31.3,
        min_humidity: 42.2,
        max_humidity: 63.6,
        min_ph: 3.8,
        max_ph: 9.0,
        min_rainfall: 30.2,
        max_rainfall: 69.4,
    },
    {
        label: 'mungbean',
        min_N: 14.2,
        max_N: 39.6,
        min_P: 37.9,
        max_P: 56.2,
        min_K: 15.8,
        max_K: 25.4,
        min_temperature: 27.3,
        max_temperature: 29.4,
        min_humidity: 80.5,
        max_humidity: 88.9,
        min_ph: 5.9,
        max_ph: 6.6,
        min_rainfall: 37.5,
        max_rainfall: 58.6,
    },
    {
        label: 'muskmelon',
        min_N: 83.3,
        max_N: 116.6,
        min_P: 5.8,
        max_P: 30.4,
        min_K: 48.3,
        max_K: 54.7,
        min_temperature: 27.2,
        max_temperature: 28.7,
        min_humidity: 91.4,
        max_humidity: 92.8,
        min_ph: 6.0,
        max_ph: 6.2,
        min_rainfall: 20.6,
        max_rainfall: 28.7,
    },
    {
        label: 'orange',
        min_N: 17.2,
        max_N: 37.8,
        min_P: 6.7,
        max_P: 29.1,
        min_K: 6.1,
        max_K: 13.1,
        min_temperature: 11.2,
        max_temperature: 33.1,
        min_humidity: 91.7,
        max_humidity: 93.2,
        min_ph: 6.1,
        max_ph: 6.7,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    {
        label: 'papaya',
        min_N: 33.0,
        max_N: 57.5,
        min_P: 46.4,
        max_P: 69.3,
        min_K: 46.0,
        max_K: 55.4,
        min_temperature: 23.5,
        max_temperature: 42.4,
        min_humidity: 91.6,
        max_humidity: 93.3,
        min_ph: 6.0,
        max_ph: 6.2,
        min_rainfall: 39.9,
        max_rainfall: 98.6,
    },
    {
        label: 'pigeonpeas',
        min_N: 19.2,
        max_N: 39.3,
        min_P: 55.4,
        max_P: 79.3,
        min_K: 16.8,
        max_K: 24.9,
        min_temperature: 19.1,
        max_temperature: 36.2,
        min_humidity: 30.9,
        max_humidity: 68.3,
        min_ph: 4.4,
        max_ph: 7.3,
        min_rainfall: 92.7,
        max_rainfall: 98.5,
    },
    {
        label: 'pomegranate',
        min_N: 20.1,
        max_N: 37.2,
        min_P: 6.2,
        max_P: 29.2,
        min_K: 36.6,
        max_K: 44.0,
        min_temperature: 18.7,
        max_temperature: 24.0,
        min_humidity: 86.2,
        max_humidity: 93.7,
        min_ph: 5.3,
        max_ph: 7.0,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    {
        label: 'rice',
        min_N: 67.1,
        max_N: 93.2,
        min_P: 39.2,
        max_P: 57.4,
        min_K: 38.8,
        max_K: 41.5,
        min_temperature: 21.6,
        max_temperature: 25.3,
        min_humidity: 83.4,
        max_humidity: 83.8,
        min_ph: 5.1,
        max_ph: 7.0,
        min_rainfall: 100.0,
        max_rainfall: 100.0,
    },
    { 
        label: 'watermelon', 
        min_N: 80, max_N: 120, 
        min_P: 5, max_P: 30, 
        min_K: 45, max_K: 55, 
        min_temperature: 24, 
        max_temperature: 26, 
        min_humidity: 80, 
        max_humidity: 89, 
        min_ph: 6, 
        max_ph: 6, 
        min_rainfall: 40, 
        max_rainfall: 59
    }
];


function getRecommendedFertilizer(crop) {
    const fertilizerRecommendations = {
        'apple': 'Urea',
        'banana': 'DAP',
        'blackgram': '14-35-14',
        'chickpea': '28-28',
        'coconut': '17-17-17',
        'coffee': '20-20',
        'cotton': '10-26-26',
        'grapes': 'Urea',
        'jute': 'DAP',
        'kidneybeans': '14-35-14',
        'lentil': '28-28',
        'maize': '17-17-17',
        'mango': '20-20',
        'mothbeans': '10-26-26',
        'mungbean': 'Urea',
        'muskmelon': 'DAP',
        'orange': '14-35-14',
        'papaya': '28-28',
        'pigeonpeas': '17-17-17',
        'pomegranate': '20-20',
        'rice': '10-26-26',
        'watermelon': 'Urea',
    };

    return fertilizerRecommendations[crop] || '17-17-17';
}

function FertilizerPage() {
    const navigate = useNavigate();
    const [soilType, setSoilType] = useState('');
    const [cropType, setCropType] = useState('');

    const handleSoilTypeChange = (event) => {
        setSoilType(event.target.value);
    };

    const handleCropTypeChange = (event) => {
        setCropType(event.target.value);
    };

    const focusEmptyFields = () => {
        const inputElements = [
            'nitrogen-fertilizer-input',
            'temp-fertilizer-input',
            'phosphorous-fertilizer-input',
            'potassium-fertilizer-input',
        ];

        for (let i = 0; i < inputElements.length; i++) {
            const inputElement = document.getElementById(inputElements[i]);
            if (!inputElement || inputElement.value === '') {
                if (inputElement) {
                    inputElement.focus();
                }
                return false;
            }
        }

        if (soilType === '' || cropType === '') {
            alert("Fill all Input fields!");
            return false;
        }

        return true;
    };

    const handleClick = () => {
        const isFieldEmpty = focusEmptyFields();
        if (!isFieldEmpty) {
            console.log("Some Inputs are empty !");
            return;
        }

        // Get the values of all text fields
        const nitrogenValue = parseFloat(document.getElementById('nitrogen-fertilizer-input').value);
        const tempValue = parseFloat(document.getElementById('temp-fertilizer-input').value);
        const phosphorousValue = parseFloat(document.getElementById('phosphorous-fertilizer-input').value);
        const potassiumValue = parseFloat(document.getElementById('potassium-fertilizer-input').value);
        const humidityValue = parseFloat(document.getElementById('humidity-fertilizer-input').value);
        const moistureValue = parseFloat(document.getElementById('moisture-fertilizer-input').value);

        // Check if the Input values are in required ranges for crops
        const selectedCrop = cropType;
        const cropDataEntry = cropData.find(crop => crop.label === selectedCrop);

        if (!cropDataEntry) {
            window.alert("Invalid crop selected!");
            return;
        }

        const {
            min_N,
            max_N,
            min_P,
            max_P,
            min_K,
            max_K,
            min_temperature,
            max_temperature,
            min_humidity,
            max_humidity,
        } = cropDataEntry;
        

        if (
            nitrogenValue < min_N || nitrogenValue > max_N ||
            phosphorousValue < min_P || phosphorousValue > max_P ||
            potassiumValue < min_K || potassiumValue > max_K ||
            tempValue < min_temperature || tempValue > max_temperature ||
            humidityValue < min_humidity || humidityValue > max_humidity
        ) {
            console.log("Input values are out of recommended range for the selected crop!");
        }

        const progressBar = document.querySelector('.fertilizer-progress-bar');
        progressBar.style.display = 'block';
        progressBar.style.visibility = 'visible';
        console.log("Fertilizer Prediction in progress...");
        console.log(cropDataEntry);

var payload={
    Temparature: tempValue,
    Humidity: humidityValue,
    Moisture: moistureValue,
    Soil_Type: soilType,
    Crop_Type: cropType,
    Nitrogen:nitrogenValue,
    Potassium: potassiumValue,
    Phosphorous: phosphorousValue
}
console.log(payload);
axios.post("http://localhost:5000/predict-fert",payload).then((response) => {
    console.log(response.data);
    

        const recommendedFertilizer = response.data.predicted_fertilizer;
        navigate("/fertilizer_result", { state: { predicted_fertilizer: recommendedFertilizer } });
    });
    }
    return (
        <>
            <Header />
            <LinearProgress style={{ visibility: 'hidden', display: 'none' }} className="fertilizer-progress-bar" />
            <p className="fertilizer-p">Input information about your soil characteristics and the crop you're cultivating to discover the optimal <b>FERTILIZER</b> for your farm 👩‍🌾🌽🚜</p>
            <div className="fertilizer-container">
                <TextField id="nitrogen-fertilizer-input" label="Ratio of Nitrogen" variant="outlined" color="success" type="number" />
                <TextField id="temp-fertilizer-input" label="Temperature in Celsius" variant="outlined" color="success" type="number" />
                <FormControl fullWidth>
                    <InputLabel id="soil-select-field">Type of Soil</InputLabel>
                    <Select
                        labelId="soil-select-field"
                        id="soil-select-field"
                        value={soilType}
                        label="Type of Soil"
                        color="success"
                        onChange={handleSoilTypeChange}
                    >
                        <MenuItem value={"Sandy"}>Sandy</MenuItem>
                        <MenuItem value={"Loamy"}>Loamy</MenuItem>
                        <MenuItem value={"Black"}>Black</MenuItem>
                        <MenuItem value={"Red"}>Red</MenuItem>
                        <MenuItem value={"Clayey"}>Clayey</MenuItem>
                    </Select>
                </FormControl>
                <FormControl fullWidth>
                    <InputLabel id="crop-select-field">Type of Crop</InputLabel>
                    <Select
                        labelId="crop-select-field"
                        id="crop-select-field"
                        value={cropType}
                        label="Type of Crop growing"
                        color="success"
                        onChange={handleCropTypeChange}
                    >
                        {cropData.map(crop => (
                            <MenuItem key={crop.label} value={crop.label}>{crop.label}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <TextField id="phosphorous-fertilizer-input" label="Ratio of Phosphorous" variant="outlined" color="success" type="number" />
                <TextField id="humidity-fertilizer-input" label="% of Humidity" variant="outlined" color="success" type="number" />
                <TextField id="potassium-fertilizer-input" label="Ratio of Potassium" variant="outlined" color="success" type="number" />
                <TextField id="moisture-fertilizer-input" label="Moisture in the soil" variant="outlined" color="success" type="number" />
                <button className="predict_fertilizer_btn" onClick={handleClick}>PREDICT</button>
            </div>
        </>
    );
}

export default FertilizerPage;