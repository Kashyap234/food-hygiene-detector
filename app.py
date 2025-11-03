import streamlit as st
from PIL import Image
import os
from hygiene_detector import HygieneDetector
import time

# Page config
st.set_page_config(
    page_title="Food Hygiene Detector",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .score-excellent { background-color: #d4edda; color: #155724; }
    .score-good { background-color: #d1ecf1; color: #0c5460; }
    .score-fair { background-color: #fff3cd; color: #856404; }
    .score-poor { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍽️ Food Hygiene Detector</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Food Safety Analysis using Multimodal LLMs")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    provider = st.selectbox(
        "Choose LLM Provider",
        ["openai", "gemini", "anthropic"],
        help="Select the AI model to use for analysis"
    )
    
    st.info("""
    **Provider Info:**
    - **OpenAI**: Most accurate, requires paid API
    - **Gemini**: Free tier available
    - **Anthropic**: Good alternative
    """)
    
    st.markdown("---")
    st.header("📊 About")
    st.write("""
    This app uses cutting-edge multimodal AI to:
    - Detect contaminants (hair, insects, etc.)
    - Assess food quality
    - Evaluate hygiene standards
    - Provide safety recommendations
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the food item"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file temporarily
        temp_path = os.path.join("images", "temp_upload.jpg")
        os.makedirs("images", exist_ok=True)
        
        # Convert image to RGB if necessary (for JPEG compatibility)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the white background
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(temp_path, 'JPEG', quality=95)

with col2:
    st.header("🔍 Analysis Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Analyze Hygiene", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing the image..."):
                try:
                    # Initialize detector
                    detector = HygieneDetector(provider=provider)
                    
                    # Perform analysis
                    result = detector.analyze(temp_path)
                    
                    if result["success"]:
                        # Display score
                        score = result["parsed"]["hygiene_score"]
                        risk = result["parsed"]["risk_level"]
                        
                        # Check if this is a N/A response (not food image)
                        if score == "N/A" or risk == "N/A":
                            st.warning("⚠️ **Unable to Analyze**")
                            st.info("The model detected that this image does not contain food or cannot be properly assessed for hygiene. Please upload a clear image of food.")
                        elif isinstance(score, int):
                            # Determine score class
                            if score >= 8:
                                score_class = "score-excellent"
                                emoji = "✅"
                            elif score >= 6:
                                score_class = "score-good"
                                emoji = "👍"
                            elif score >= 4:
                                score_class = "score-fair"
                                emoji = "⚠️"
                            else:
                                score_class = "score-poor"
                                emoji = "❌"
                            
                            st.markdown(f"""
                            <div class="score-box {score_class}">
                                {emoji} Hygiene Score: {score}/10
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display risk level
                            st.markdown("---")
                            risk_colors = {
                                "LOW": "🟢",
                                "MEDIUM": "🟡",
                                "HIGH": "🟠",
                                "CRITICAL": "🔴"
                            }
                            st.subheader(f"{risk_colors.get(risk, '⚪')} Risk Level: {risk}")
                        else:
                            # Score not detected (but not explicitly N/A)
                            st.warning("⚠️ Score could not be extracted from the analysis.")
                            if risk != "UNKNOWN":
                                st.markdown("---")
                                risk_colors = {
                                    "LOW": "🟢",
                                    "MEDIUM": "🟡",
                                    "HIGH": "🟠",
                                    "CRITICAL": "🔴"
                                }
                                st.subheader(f"{risk_colors.get(risk, '⚪')} Risk Level: {risk}")
                        
                        # Full analysis
                        st.markdown("---")
                        st.subheader("📋 Detailed Analysis")
                        st.write(result["raw_response"])
                        
                        # Download option
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Report",
                            data=result["raw_response"],
                            file_name=f"hygiene_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                    else:
                        st.error(f"❌ Analysis failed: {result['error']}")
                        st.info("💡 Please check your API key in the .env file")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("""
                    **Common Issues:**
                    1. Missing API key in .env file
                    2. Invalid API key
                    3. Insufficient API credits
                    4. Network connection issues
                    """)
    else:
        st.info("👆 Please upload an image to start analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit & Multimodal LLMs | For educational purposes</p>
</div>
""", unsafe_allow_html=True)