
import React from 'react';
import { Link } from 'react-router-dom';
import "./Header.css"

const Header = () => {
    return (
        <div className="header">
            <div className="logo"> <Link to="/"> 🍃 SoilMinds </Link> </div>
            <ul>
                <li className="crop-header-option">
                    <Link to="/crop">Crop Recommendation </Link>
                </li>
                <li className="fertilizer-header-option">
                    <Link to="/fertilizer"> Fertilizer Recommendation </Link>
                </li>
            </ul>
        </div>
    );
};

export default Header;
