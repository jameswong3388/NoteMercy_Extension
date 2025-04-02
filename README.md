<img width="1728" alt="Screenshot 2025-04-01 at 9 18 44 AM" src="https://github.com/user-attachments/assets/648d9bba-d21c-4b6b-aa26-5934839e5f33" />

## Project Setup

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

2. Create a virtual environment (recommended) or you can use your own setup:
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
   
### Additional Notes

- Make sure you have Node.js (16.x or later) installed for the frontend
- Python 3.8+ is required for the backend
- Both frontend and backend servers need to be running for the application to work properly

## Datasets

The project includes sample test images available in the `/backend/atest` folder. The test dataset consists of various handwriting styles:

- **Print Writing**: `print.jpg`, `print2.png`
- **Cursive Writing**: `cursive2.png`, `cursive3.jpg`, `cursive4.png`
- **Italic Writing**: `italic.jpg`, `italic2.png`, `italic3.png`, `italic4.png`, `italic5.png`
- **Calligraphic Writing**: `calligraphic.png`, `calligraphic2.png`
- **Shorthand Writing**: `shorthand1.png` through `shorthand5.png`
- **Block Writing**: `block1.png`, `block2.png`, `block3.png`

These images serve as test cases for the handwriting recognition system and demonstrate the system's capability to handle various writing styles.