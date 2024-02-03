import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { HashLoader } from "react-spinners";
import Container from '@mui/material/Container';
import Header from "../header/Header.jsx";
import axios from "axios";

const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'justify',
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
    color: '#333',
    fontFamily: 'Open Sans, sans-serif',
};

const headerStyle = {
    backgroundColor: '#4CAF50',
    color:'rgb(51, 51, 51)',
    padding: '20px',
    textAlign: 'center',
    width: '100vw',
};

const mainStyle = {
    padding: '20px',
    display: 'grid',
    gap: '20px',
    textAlign: 'justify',
};

const buttonStyle = {
    backgroundColor: '#4CAF50',
    color: 'white',
    padding: '12px 24px',
    fontSize: '18px',
    borderRadius: '5px',
    cursor: 'pointer',
    border: 'none',
    marginBottom: '2rem'
};

const sectionHeadingStyle = {
    fontSize: '32px',
    color: '#4CAF50', 
    marginBottom: '20px',
    textAlign: 'justify',
};

function HomePage(props) {
    const navigate = useNavigate();

    return (
        <>
            <Header className="header" />
            <div style={containerStyle}>
                <header style={headerStyle}>
                    <h1 style={{ fontSize: '48px', color: 'rgb(51, 51, 51)' }}>Welcome to SoilMinds</h1>
                    <p style={{ fontSize: '20px' }}>Nurturing Tomorrow's Harvest Today 🌱</p>
                </header>
                <main style={mainStyle}>
                    <section>
                        <h2 style={sectionHeadingStyle}>Explore the Future of Farming with SoilMinds</h2>
                        <p>At SoilMinds, we're redefining agriculture by blending technology with the art of cultivation. Step into a world where your farm's potential is maximized, and every harvest becomes a success story.</p>
                    </section>
                    <section>
                        <h2 style={sectionHeadingStyle}>Our Core Offerings:</h2>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            <li><strong>Precision Crop Recommendations:</strong> Unleash the full potential of your fields with our precise crop recommendations. SoilMinds analyzes the nitrogen, phosphorus, and potassium ratios, ensuring your crops receive the tailored care they deserve.</li>
                            <li><strong>Climate Intelligence:</strong> Understand your farm's environment like never before. SoilMinds considers crucial factors like humidity, temperature, pH value, and rainfall data to provide you with insights that elevate your farming strategies.</li>
                            <li><strong>Fertilizer Insights:</strong> Make informed decisions with our fertilizer recommendations. SoilMinds goes beyond suggestions and provides actionable insights, ensuring your crops receive the nutrients they need for optimal growth.</li>
                        </ul>
                    </section>
                    <section>
                        <h2 style={sectionHeadingStyle}>Why SoilMinds?</h2>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            <li><strong>Innovation at Your Fingertips:</strong> Embrace the latest advancements in agricultural technology. SoilMinds puts innovation directly in your hands, transforming the way you nurture your crops.</li>
                            <li><strong>User-Friendly Experience:</strong> No need to be a tech expert. SoilMinds boasts an intuitive interface designed for every farmer. Navigate effortlessly and harness the power of technology with ease.</li>
                            <li><strong>Sustainable Agriculture:</strong> Join a community of forward-thinking farmers committed to sustainable practices. SoilMinds isn't just a tool; it's a commitment to a brighter, more sustainable future for agriculture.</li>
                        </ul>
                    </section>
                    <section>
                        <h2 style={sectionHeadingStyle}>Ready to Transform Your Farming Experience?</h2>
                        <p>Embark on a journey where technology meets the soil. Click the "Get Started" button below and open the door to a new era of farming.</p>
                        <p>Welcome to SoilMinds, where every seed planted is a step towards a thriving and sustainable harvest.</p>
                    </section>
                </main>
                <button onClick={() => navigate("/crop")} style={buttonStyle}>Get Started Now 🌱</button>
                <div className="container">
                    {props.children}
                </div>
            </div>
        </>
    );
}

export function ModelLoader() {
    const [isLoading, setIsLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        axios.get("http://127.0.0.1:5000").then((response) => {
            console.log(response.data);
        }
        );
        
        setTimeout(() => {
            setIsLoading(false);
        }, 2000);
    }, []);

    if (isLoading) {
        return (
            <div style={containerStyle}>
                <Container maxWidth="md">
                    <HashLoader
                        className="spinner"
                        color={"#4CAF50"}
                        loading={true}
                        cssOverride={{ display: "block", margin: "0 auto" }}
                        size={80}
                        aria-label="Loading..."
                        data-testid="loader"
                    />
                </Container>
            </div>
        );
    }

    return (
        <div>
            <HomePage />
        </div>
    );
}
