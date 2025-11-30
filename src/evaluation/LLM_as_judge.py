# step2_evaluate_with_gpt.py

import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
import dotenv
dotenv()
# CONFIGURATION
client = OpenAI()  
RESPONSES_DIR = "data/model_responses/replay"
OUTPUT_DIR = "data/evaluation_results/replay/results_gpt_judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPT model to use as judge
JUDGE_MODEL = "gpt-4o"  

EVALUATION_PROMPT_TEMPLATE = """You are a strict, impartial constitutional law professor acting as an evaluator.
Your task is to evaluate a CANDIDATE ANSWER against a GOLD ANSWER (Ground Truth) for a specific legal question.

### INPUT DATA:
- Question Difficulty: {difficulty} (E=Easy, M=Medium, H=Hard)
- Question: {question}
- GOLD ANSWER: {gold_answer}
- CANDIDATE ANSWER: {candidate_answer}

### EVALUATION RULES:
1. **Objective Truth:** The Gold Answer is your absolute source of truth. If the Candidate contradicts the Gold Answer on a holding, statute, or principle, it is WRONG.
2. **No Style Bias:** Do NOT award points for eloquence, politeness, or length. A concise, correct answer is better than a verbose, vague one.
3. **Handling Hallucinations:** If the Candidate cites a case, date, or principle that DOES NOT EXIST or is legally incorrect (even if not explicitly in the Gold Answer), penalize `factual_correctness` heavily.
4. **Handling Refusals/Gibberish:** If the Candidate refuses to answer or outputs irrelevant text, score all numeric fields as 0.

### SCORING RUBRIC:
**1. Factual Correctness (0-5)**
- 5: Factually perfect. No errors in case names, dates, or holdings.
- 4: Correct, but with very minor details missed (e.g., slightly off date).
- 3: Generally correct legal principle, but applies it loosely or makes a minor error.
- 2: Contains a significant legal error or hallucination.
- 1: Mostly incorrect or fundamentally misunderstands the law.
- 0: Completely wrong, irrelevant, or refuses to answer.

**2. Coverage of Key Points (0-5)**
- 5: Identifying all components (Holding + Rationale + Key Dissent/Nuance if applicable).
- 3-4: Identifies the main holding but misses the rationale or secondary prong.
- 1-2: Misses the central holding entirely but mentions relevant keywords.
- 0: Irrelevant.

**3. Nuance & Reasoning (0-3)** (Context-Dependent)
- *For Hard (H) Questions:* Did the model apply the doctrine to the hypothetical correctly? Did it synthesize the two conflicting cases?
- 3: Sophisticated synthesis; accurately handles the "grey area."
- 2: Logical application, but misses the deeper conflict.
- 1: Surface-level recitation of rules without proper application.
- 0: Failed reasoning or logical contradiction.

**4. Expression (0-2)**
- 2: Clear, precise legal terminology.
- 1: Understandable but uses layperson terms instead of terms of art.
- 0: Incoherent.

### OUTPUT FORMAT:
Return valid JSON only. Do not include markdown formatting (like ```json).
{{
  "difficulty": "{difficulty}",
  "factual_correctness": <int 0-5>,
  "coverage_of_key_points": <int 0-5>,
  "nuance_and_reasoning": <int 0-3>,
  "expression_and_clarity": <int 0-2>,
  "overall_score": <float 0-10, weighted towards accuracy>,
  "verdict": "<Excellent|Good|Mixed|Poor|Fail>",
  "hallucination_detected": <bool>,
  "short_feedback": "<Critical assessment of *why* it lost points. Quote specific errors if present. Max 2 sentences.>"
}}"""


# HELPER FUNCTIONS
def evaluate_response(question, gold_answer, candidate_answer, difficulty):
    """Send a single evaluation request to GPT."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        difficulty=difficulty,
        question=question,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict constitutional law evaluator. Always return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown formatting if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # Parse JSON
            evaluation = json.loads(result_text)
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "difficulty": difficulty,
                    "factual_correctness": 0,
                    "coverage_of_key_points": 0,
                    "nuance_and_reasoning": 0,
                    "expression_and_clarity": 0,
                    "overall_score": 0.0,
                    "verdict": "Fail",
                    "hallucination_detected": False,
                    "short_feedback": "Evaluation failed due to JSON parsing error.",
                    "error": str(e)
                }
            time.sleep(1)
        except Exception as e:
            print(f"  ⚠ Error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {
                    "difficulty": difficulty,
                    "factual_correctness": 0,
                    "coverage_of_key_points": 0,
                    "nuance_and_reasoning": 0,
                    "expression_and_clarity": 0,
                    "overall_score": 0.0,
                    "verdict": "Fail",
                    "hallucination_detected": False,
                    "short_feedback": f"Evaluation failed: {str(e)}",
                    "error": str(e)
                }
            time.sleep(1)


def main():
    print("Starting GPT-based evaluation...")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Input Directory: {RESPONSES_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Get all response files (filtering for your specific files)
    response_files = sorted([f for f in os.listdir(RESPONSES_DIR) if f.startswith('phi_1.5_step')])
    
    if not response_files:
        print(f"No response files found in {RESPONSES_DIR}")
        return
    
    print(f"\nFound {len(response_files)} response files to evaluate\n")
    print(response_files)
    
    # Process each file
    for response_file in response_files:
        print(f"Evaluating: {response_file}")
        
        input_path = os.path.join(RESPONSES_DIR, response_file)
        output_path = os.path.join(OUTPUT_DIR, response_file.replace('responses', 'evaluations'))
        
        # --- ROBUST JSON LOADING LOGIC ---
        responses = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                decoder = json.JSONDecoder()
                pos = 0
                
                while pos < len(content):
                    # Skip any whitespace (newlines/spaces) between objects
                    while pos < len(content) and content[pos].isspace():
                        pos += 1
                    
                    if pos >= len(content):
                        break
                    
                    try:
                        # raw_decode parses one object and returns the object + the end index
                        obj, end_pos = decoder.raw_decode(content[pos:])
                        responses.append(obj)
                        pos += end_pos  # Move pointer to the end of this object
                    except json.JSONDecodeError as e:
                        print(f"  Warning: JSON decode error at position {pos}: {e}")
                        # Move forward one char to attempt recovery, or break if hopeless
                        pos += 1
        except Exception as e:
            print(f"CRITICAL ERROR reading file {input_path}: {e}")
            continue
        # ---------------------------------
        
        print(f"Loaded {len(responses)} responses")
        
        if not responses:
            print(f"Skipping empty or malformed file: {response_file}")
            continue

        # Evaluate each response
        evaluated_results = []
        for response_data in tqdm(responses, desc="Evaluating"):
            evaluation = evaluate_response(
                question=response_data['question'],
                gold_answer=response_data['gold_answer'],
                candidate_answer=response_data['candidate_answer'],
                difficulty=response_data['difficulty']
            )
            
            # Combine original data with evaluation
            result = {**response_data, "evaluation": evaluation}
            evaluated_results.append(result)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Save evaluations
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for result in evaluated_results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Saved evaluations to {output_path}")
        
        # Print summary statistics
        if evaluated_results:
            avg_score = sum(r['evaluation']['overall_score'] for r in evaluated_results) / len(evaluated_results)
            hallucinations = sum(1 for r in evaluated_results if r['evaluation'].get('hallucination_detected', False))
            
            print(f"\n Average Score: {avg_score:.2f}/10")
            print(f" Hallucinations Detected: {hallucinations}/{len(evaluated_results)}")
    
    print(f"All Results saved to: {OUTPUT_DIR}")
if __name__ == "__main__":
    main()