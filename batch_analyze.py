import os
import json
from datetime import datetime
from hygiene_detector import HygieneDetector
from pathlib import Path

def batch_analyze(input_folder="images", output_folder="output", provider="openai"):
    """
    Analyze multiple images in batch
    
    Args:
        input_folder: Folder containing images to analyze
        output_folder: Folder to save results
        provider: LLM provider to use
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize detector
    print(f"Initializing {provider.upper()} detector...")
    detector = HygieneDetector(provider=provider)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_folder) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to analyze\n")
    
    results = []
    
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, filename)
        print(f"[{idx}/{len(image_files)}] Analyzing {filename}...")
        
        try:
            result = detector.analyze(image_path)
            
            if result["success"]:
                score = result["parsed"]["hygiene_score"]
                risk = result["parsed"]["risk_level"]
                
                print(f"  ✓ Score: {score}/10 | Risk: {risk}")
                
                # Save individual result
                results.append({
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),
                    "score": score,
                    "risk_level": risk,
                    "analysis": result["raw_response"]
                })
            else:
                print(f"  ✗ Failed: {result['error']}")
                results.append({
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),
                    "error": result["error"]
                })
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        print()
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"batch_results_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_file}")
    
    # Generate summary
    successful = [r for r in results if "score" in r]
    failed = [r for r in results if "error" in r]
    
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total Images: {len(image_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        scores = [r["score"] for r in successful if isinstance(r["score"], int)]
        if scores:
            print(f"\nAverage Hygiene Score: {sum(scores)/len(scores):.1f}/10")
            print(f"Highest Score: {max(scores)}/10")
            print(f"Lowest Score: {min(scores)}/10")
        
        risk_counts = {}
        for r in successful:
            risk = r.get("risk_level", "UNKNOWN")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print("\nRisk Distribution:")
        for risk, count in sorted(risk_counts.items()):
            print(f"  {risk}: {count} images")
    
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch analyze food images")
    parser.add_argument("--input", default="images", help="Input folder with images")
    parser.add_argument("--output", default="output", help="Output folder for results")
    parser.add_argument("--provider", default="openai", 
                        choices=["openai", "gemini", "anthropic"],
                        help="LLM provider to use")
    
    args = parser.parse_args()
    
    batch_analyze(args.input, args.output, args.provider)