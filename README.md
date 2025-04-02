# NoteMercy Extension

<img width="1728" alt="Screenshot 2025-04-01 at 9 18 44 AM" src="https://github.com/user-attachments/assets/648d9bba-d21c-4b6b-aa26-5934839e5f33" />

## Overview

NoteMercy Extension is a sophisticated handwriting analysis and recognition system that can identify and analyze various styles of handwriting. The system provides detailed analysis of different handwriting characteristics including print, cursive, italic, calligraphic, shorthand, and block writing styles.

## Features

- **Multiple Handwriting Style Analysis**
  - Print Writing
  - Cursive Writing
  - Italic Writing
  - Calligraphic Writing
  - Shorthand Writing
  - Block Writing

- **Advanced Analysis Components**
  - Angularity Analysis
  - Aspect Ratio Analysis
  - Loop Detection
  - Stroke Width Variation
  - Curvature Analysis
  - Letter Spacing Analysis
  - And more specialized analyzers for each writing style

## Project Structure

```
NoteMercy_Extension/
├── frontend/           # Next.js frontend application
├── backend/           # Python FastAPI backend
│   ├── atest/        # Test datasets
│   ├── lib_py/       # Analysis libraries
│   └── api.py        # Main API server
└── README.md
```

## Prerequisites

- Node.js 16.x or later
- Python 3.8 or later
- npm or yarn package manager

## Installation

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

The frontend will be available at `http://localhost:3000`

### Backend Setup (Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python api.py
   ```

## Test Datasets

The project includes sample test images in the `/backend/atest` folder demonstrating various handwriting styles:

- **Print Writing**: `print.jpg`, `print2.png`
- **Cursive Writing**: `cursive2.png`, `cursive3.jpg`, `cursive4.png`
- **Italic Writing**: `italic.jpg`, `italic2.png`, `italic3.png`, `italic4.png`, `italic5.png`
- **Calligraphic Writing**: `calligraphic.png`, `calligraphic2.png`
- **Shorthand Writing**: `shorthand1.png` through `shorthand5.png`
- **Block Writing**: `block1.png`, `block2.png`, `block3.png`

These images serve as test cases for the handwriting recognition system and demonstrate the system's capability to handle various writing styles.

## Development Notes

- Both frontend and backend servers must be running for the application to function properly
- The backend API is built with FastAPI and includes CORS support for frontend communication
- The frontend is built with Next.js and uses modern web technologies including Tailwind CSS
- The system implements various analyzers for different aspects of handwriting analysis

## Contributing

Feel free to submit issues and enhancement requests!