import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export interface RLWeights {
  realDocBias: number;         // Offset added dynamically for genuine document classes
  strictnessModifier: number;  // Multiplier for suspicious marker deductions
  overallCalibration: number;  // Global score shift modifier
  feedbackCount: number;       // Training epochs/feedback loop logs
  learningRate: number;        // Gradient descent step scaler
}

export interface AnalysisResult {
  trust_score: number;
  classification: "Likely Real" | "Suspicious" | "Likely Fake";
  reasons: string[];
  suggestions: string[];
  heatmap_regions: { x: number; y: number; width: number; height: number; reason: string }[];
  explanation: string;
}

export async function analyzeDocument(
  imageBase64: string, 
  mimeType: string, 
  rlWeights?: RLWeights
): Promise<AnalysisResult> {
  const realBias = rlWeights?.realDocBias ?? 0;
  const strictness = rlWeights?.strictnessModifier ?? 1.0;
  const calibration = rlWeights?.overallCalibration ?? 0;

  const prompt = `
    You are a world-class forensic document analyst. Analyze the provided document for signs of deepfake, AI generation, or manual tampering.
    
    CRITICAL: If the image quality is too low to perform a reliable analysis, you MUST mention this in the 'reasons' and 'explanation' and lower the trust score significantly (below 50).
    
    Follow this EXACT pipeline:
    1. Metadata Check: Look for signs of editing software or missing standard metadata patterns visible in the image.
    2. Image Forensics: Detect abnormal edges, pixelation, or inconsistent lighting/textures.
    3. Vision LLM Analysis: Use advanced vision-language processing to detect semantic inconsistencies (e.g., date mismatches, logical errors in text), AI-generated text artifacts (e.g., character bleeding, inconsistent kerning), and structural document anomalies that traditional OCR might miss.
    4. Biometric Verification: If a face is present, perform a deep-level biometric check for "uncanny valley" effects, skin texture inconsistencies, and lighting/shadow mismatches that indicate a synthetic or swapped face.
    
    REINFORCEMENT LEARNING DIRECTIVE (CRITICAL):
    You have been trained through user-assigned reinforcement rewards to calibrate your parameters to solve false-positives/false-negatives. Apply the following calculated policy biases in your weights:
    - Real Document Bias: ${realBias} points. (If the document is genuine or has only very trivial layout/scan noise, BOOST the score, or prevent excessive deduction, using this weight. Users reported the model can be too strict on authentic documents.)
    - Strictness Multiplier: ${strictness}x. (Multiply all standard deduction weights, such as compression shadows or edge noise, by this factor.)
    - Global Trust Calibration: ${calibration} points. (Offset your finalized trust score value by adding exactly this calibration value to prevent underestimation or false alarms.)
    
    Scoring Protocol:
    - Start at 100.
    - Calculate deductions:
        * Deduct 20-40 points for clear manipulation (cloning, healing, text replacement).
        * Deduct 15-25 points for suspicious metadata or low resolution.
        * Deduct 10-20 points for inconsistent lighting or shadows.
    - Scale these calculated deductions by the Strictness Multiplier (${strictness}x).
    - Add the Real Document Bias (${realBias}) if the document overall appears to be an authentic scan/photo.
    - Add the Global Trust Calibration (${calibration}) to the final calculated score.
    - Ensure your final returned trust_score is constrained strictly between 0 and 100.
    - If multiple critical forgery artifacts are found, the ultimate score MUST be below 40 (Likely Fake).
    
    Return the result in JSON format with the following structure:
    {
      "trust_score": number,
      "classification": string,
      "reasons": string[],
      "suggestions": string[],
      "heatmap_regions": [{"x": percentage_x, "y": percentage_y, "width": percentage_w, "height": percentage_h, "reason": string}],
      "explanation": string
    }
    
    Note: heatmap_regions should use percentages (0-100) for coordinates and dimensions relative to the image size. 
    Ensure the regions accurately highlight the suspicious areas described in the reasons.
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [
        {
          parts: [
            { text: prompt },
            { inlineData: { data: imageBase64.split(",")[1] || imageBase64, mimeType } }
          ]
        }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            trust_score: { type: Type.NUMBER },
            classification: { type: Type.STRING },
            reasons: { type: Type.ARRAY, items: { type: Type.STRING } },
            suggestions: { type: Type.ARRAY, items: { type: Type.STRING } },
            heatmap_regions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  x: { type: Type.NUMBER },
                  y: { type: Type.NUMBER },
                  width: { type: Type.NUMBER },
                  height: { type: Type.NUMBER },
                  reason: { type: Type.STRING }
                },
                required: ["x", "y", "width", "height", "reason"]
              }
            },
            explanation: { type: Type.STRING }
          },
          required: ["trust_score", "classification", "reasons", "suggestions", "heatmap_regions", "explanation"]
        }
      }
    });

    const result = JSON.parse(response.text || "{}");
    return result as AnalysisResult;
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    throw new Error("Failed to analyze document. Please try again.");
  }
}
