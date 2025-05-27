import logging
import time
import torch
import re

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Manages interactions with the Large Language Model (LLM) for generating responses.
    Includes prompt formatting, inference, and response cleaning.
    """

    def __init__(self, config, llm_model, tokenizer):
        """
        Initializes the LLMHandler.
        """
        self.config = config
        self.llm = llm_model
        self.tokenizer = tokenizer

    def generate_response(self, system_prompt, user_prompt):
        """
        Generates a response from the LLM given a system prompt and a user prompt.
        Includes error handling, retry logic, and output cleaning.
        """
        max_retries = 2 
        for attempt in range(max_retries + 1): 
            try: 
                full_prompt = f"""{system_prompt}

    USER INPUT: {user_prompt}

    INSTRUCTIONS: Respond ONLY to the specific user query above. Be concise and direct.
    - Do not add any meta-instructions or imaginary dialogue
    - Do not include phrases like 'User:' or 'Assistant:' in your response
    - Do not continue the conversation with follow-up questions
    - Do not try to predict what the user might ask next
    - Keep responses brief and to the point

    YOUR RESPONSE:""" 
                
                prompt_preview = full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt
                logger.debug(f"LLM inference prompt (truncated): {prompt_preview}")
                
                start_time = time.time()
                
                inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, 
                                    max_length=4096)  
                if self.config["use_gpu"]:
                    inputs = {k: v.to("cuda") for k, v in inputs.items()} 

                # Generate response using the LLM.
                with torch.no_grad():
                    encoding_time = time.time() - start_time
                    start_generation = time.time()

                    output = self.llm.generate(
                        **inputs,
                        max_new_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"] * 0.7,
                        do_sample=True,
                        repetition_penalty=1.2,  # Applies a penalty to discourage repetitive phrases.
                        no_repeat_ngram_size=3,  # Prevents the model from repeating n-grams of size 3 or more.
                    )
                    generation_time = time.time() - start_generation

                # Decode the generated token IDs back into human-readable text.
                full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                decoding_time = time.time() - start_generation - generation_time
                
                # Extract just the response part (after "YOUR RESPONSE:") from the full output.
                response_marker = "YOUR RESPONSE:"
                if response_marker in full_output:
                    response = full_output.split(response_marker, 1)[1].strip()
                else: # Fallback if the marker is not found (e.g., if LLM didn't follow the prompt exactly).
                    response = full_output[len(full_prompt):].strip() # Assumes the response starts immediately after the input prompt.
                
                # Advanced cleaning with more robust handling of patterns to remove unwanted LLM artifacts.
                cleaning_patterns = [ # List of regex patterns and their replacements.
                    (r'^assistant[:\s]', '', re.IGNORECASE), 
                    (r'^user[:\s]', '', re.IGNORECASE), 
                    (r'^(User|Assistant|System)[:]\s*', '', re.IGNORECASE),
                    (r'^\s*I should respond with\s*', '', re.IGNORECASE), 
                    (r'^\s*Respond with\s*', '', re.IGNORECASE), 
                    (r'User:.*$', '', re.DOTALL), 
                    (r'This helps the user.*$', '', re.DOTALL),
                    (r'(User|Assistant|System):.*$', '', re.DOTALL), 
                    (r'(Can I|Would you|Do you|Is there).*\?+\s*$', '', re.IGNORECASE|re.DOTALL),
                    (r'(Let me know|Please tell me).*\?+\s*$', '', re.IGNORECASE|re.DOTALL), 
                    (r'\s+([.!?])', r'\1', 0),  # Fixes trailing whitespace before punctuation (e.g., "hello . " -> "hello.").
                    (r'([^.!?])\s*$', r'\1.', 0),
                ]
                for pattern, repl, flags in cleaning_patterns:
                    response = re.sub(pattern, repl, response, flags=flags)
                
                total_time = time.time() - start_time
                logger.debug(f"LLM performance: total={total_time:.2f}s (encoding={encoding_time:.2f}s, "
                            f"generation={generation_time:.2f}s, decoding={decoding_time:.2f}s)")
                
                # Log the length of the generated response.
                logger.debug(f"Generated response of {len(response)} chars")
                
                return response.strip()
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"CUDA out of memory in LLM inference (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    if self.config["use_gpu"]:
                        torch.cuda.empty_cache()
                    system_prompt = "You are a helpful assistant. Respond briefly."
                    user_prompt = " ".join(user_prompt.split()[:50])
                    logger.info("Retrying with smaller prompt after CUDA OOM...")
                else: # If no retries are left.
                    return "I'm having memory issues processing your request. Could you make it shorter and try again?"
            
            except Exception as e: 
                logger.error(f"Error in LLM inference (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                if attempt < max_retries:
                    logger.info("Retrying with simpler prompt...")
                    system_prompt = "You are a helpful assistant. Please respond to the user's request."
                else:
                    return "I'm having trouble generating a response at the moment. Could you rephrase or try again?"
