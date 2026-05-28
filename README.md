# DeepGuard: Deepfake Document Detection System

DeepGuard is a production-ready web application designed to detect forged, AI-generated, or tampered documents using advanced image forensics and AI analysis.

## Features

- **Cyber-Security Theme**: Dark mode UI with neon accents and technical dashboard aesthetics.
- **Forensic Analysis Pipeline**:
  - **Metadata Check**: Signatures of editing software.
  - **Image Forensics**: Edge detection and texture inconsistency analysis.
  - **OCR Analysis**: Character spacing and font artifact detection.
  - **Face Detection**: Blurring and distortion checks for ID photos.
- **Explainable AI**: Detailed reports with trust scores, classification, and visual heatmaps.
- **Secure Uploads**: Express-based backend for handling document uploads.

## Tech Stack

- **Frontend**: React (Vite), Tailwind CSS, Framer Motion, shadcn/ui.
- **Backend**: Node.js (Express), Multer.
- **AI Engine**: Google Gemini 3 Flash (via @google/genai).

## Getting Started

1. **Environment Variables**:
   - `GEMINI_API_KEY`: Required for AI analysis.
2. **Installation**:
   - `npm install`
3. **Development**:
   - `npm run dev`
4. **Build**:
   - `npm run build`

## Analysis Protocol

The system uses a rule-based scoring system (0-100):
- **Likely Real**: > 70
- **Suspicious**: 41 - 70
- **Likely Fake**: <= 40

Visual heatmaps highlight suspicious regions for manual verification.
