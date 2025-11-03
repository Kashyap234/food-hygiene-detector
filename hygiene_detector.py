import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import re

load_dotenv()


class HygieneDetector:
    """Food hygiene detection using multimodal LLMs"""
    
    def __init__(self, provider="gemini"):
        """
        Initialize detector with chosen LLM provider
        
        Args:
            provider: 'openai', 'gemini', or 'anthropic'
        """
        self.provider = provider
        self._setup_client()
    
    def _setup_client(self):
        """Setup API client based on provider"""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o"
            
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel('gemini-2.5-flash')
            
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20240620"

    def encode_image(self, image_path):
        """Convert image to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_prompt(self):
        """Create detailed prompt for hygiene assessment"""
        return """Analyze this food image for hygiene and contamination issues.

INSPECT FOR:
1. Foreign Objects: hair, insects (flies, cockroaches, ants), lizards, rats, plastic, metal pieces
2. Food Quality: mold, discoloration, spoilage signs
3. Presentation: cleanliness of plate/container, surrounding environment
4. Preparation Issues: burnt food, undercooked items, cross-contamination signs

PROVIDE:
1. **Detected Issues**: List each contamination/issue found with confidence level (High/Medium/Low)
2. **Hygiene Score**: 0-10 scale (10 = perfectly clean, 0 = severely contaminated)
3. **Risk Level**: LOW, MEDIUM, HIGH, or CRITICAL
4. **Recommendations**: Specific actions to take
5. **Justification**: Explain the score reasoning

Be thorough but objective. If the image is unclear, mention it."""
    
    def analyze_openai(self, image_path):
        """Analyze using OpenAI GPT-4 Vision"""
        base64_image = self.encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.create_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def analyze_gemini(self, image_path):
        """Analyze using Google Gemini"""
        img = Image.open(image_path)
        response = self.client.generate_content([self.create_prompt(), img])
        return response.text
    
    def analyze_anthropic(self, image_path):
        """Analyze using Anthropic Claude"""
        base64_image = self.encode_image(image_path)
        
        # Detect image type
        with Image.open(image_path) as img:
            media_type = f"image/{img.format.lower()}"
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.create_prompt()
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text
    
    def analyze(self, image_path):
        """
        Main analysis method
        
        Args:
            image_path: Path to food image
            
        Returns:
            dict: Analysis results with score, issues, and recommendations
        """
        try:
            # Route to appropriate provider
            if self.provider == "openai":
                result = self.analyze_openai(image_path)
            elif self.provider == "gemini":
                result = self.analyze_gemini(image_path)
            elif self.provider == "anthropic":
                result = self.analyze_anthropic(image_path)
            
            # Parse the result
            parsed = self._parse_result(result)
            return {
                "success": True,
                "raw_response": result,
                "parsed": parsed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_result(self, result):
        """
        Extract structured data from LLM response using regex.
        Handles N/A cases when the model cannot analyze the image.
        """
        
        score = None
        risk_level = "UNKNOWN"
        result_lower = result.lower()
        
        # Check if model indicates it cannot analyze (not food, unclear image, etc.)
        na_indicators = [
            'n/a',
            'not applicable',
            'cannot be scored',
            'cannot be assessed',
            'does not depict food',
            'not food',
            'is not of food',
            'image is not',
            'not a food image',
            'unable to assess'
        ]
        
        is_na_response = any(indicator in result_lower for indicator in na_indicators)
        
        # --- Parse Score ---
        if is_na_response:
            # Check specifically for "Hygiene Score: N/A" pattern
            if re.search(r'hygiene score[:\s]*n/?a', result_lower):
                score = "N/A"
        
        # If not N/A, try to parse numeric score
        if score is None:
            score_match = re.search(r'\b(hygiene\s+score|score)[^0-9]*\b([0-9]|10)\b', result, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(2))
        
        # --- Parse Risk Level ---
        if is_na_response:
            # Check specifically for "Risk Level: N/A" pattern
            if re.search(r'risk level[:\s]*n/?a', result_lower):
                risk_level = "N/A"
        
        # If not N/A, try to parse actual risk level
        if risk_level not in ["N/A"]:
            risk_match = re.search(r'\b(risk level)\b[^a-zA-Z]*\b(CRITICAL|HIGH|MEDIUM|LOW)\b', result, re.IGNORECASE)
            if risk_match:
                risk_level = risk_match.group(2).upper()
            else:
                # Fallback search
                if 'critical' in result_lower:
                    risk_level = "CRITICAL"
                elif 'high' in result_lower:
                    risk_level = "HIGH"
                elif 'medium' in result_lower:
                    risk_level = "MEDIUM"
                elif 'low' in result_lower:
                    risk_level = "LOW"

        # Set defaults if still not found
        if score is None:
            score = "Not detected"
        if risk_level == "UNKNOWN":
            risk_level = "UNKNOWN"

        return {
            "hygiene_score": score,
            "risk_level": risk_level,
            "full_analysis": result
        }


# Example usage
if __name__ == "__main__":
    # Initialize detector (choose provider: 'openai', 'gemini', or 'anthropic')
    detector = HygieneDetector(provider="openai")
    
    # Analyze an image
    image_path = "images/food_sample.jpg"
    
    if os.path.exists(image_path):
        print(f"Analyzing {image_path}...")
        result = detector.analyze(image_path)
        
        if result["success"]:
            print("\n" + "="*50)
            print("HYGIENE ANALYSIS RESULTS")
            print("="*50)
            print(result["raw_response"])
            print("\n" + "="*50)
            print(f"Score: {result['parsed']['hygiene_score']}/10")
            print(f"Risk Level: {result['parsed']['risk_level']}")
            print("="*50)
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Image not found: {image_path}")
        print("Please add a food image to the 'images' folder")