import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from typing import Optional, List
import os  # Import the os module

def load_model_and_tokenizer(
    model_name_or_path: str,
    use_auth_token: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",  # Let Transformers handle device placement
    low_cpu_mem_usage: bool = True,
):
    """
    Loads the Llama 2 model and tokenizer.

    Args:
        model_name_or_path (str): The name or path of the Llama 2 model.
        use_auth_token (Optional[str], optional): Hugging Face authentication token. Defaults to None.
        torch_dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16.
        device_map (str, optional): Device mapping strategy.  "auto" lets transformers decide.
        low_cpu_mem_usage (bool): Try to use less CPU memory.

    Returns:
        tuple: (tokenizer, model)
            - tokenizer: The tokenizer for the Llama 2 model.
            - model: The Llama 2 model.  Returns None if there's an error.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_auth_token=use_auth_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            use_auth_token=use_auth_token,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None



def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    eos_token_ids: Optional[List[int]] = None,
):
    """
    Generates text using the Llama 2 model with streaming output.

    Args:
        model: The Llama 2 model.
        tokenizer: The tokenizer for the Llama 2 model.
        prompt (str): The input prompt.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
        temperature (float, optional): The temperature for sampling. Defaults to 0.7.
        top_p (float, optional): The top-p value for sampling. Defaults to 0.95.
        top_k (int, optional): The top-k value for sampling. Defaults to 50.
        repetition_penalty (float, optional): The repetition penalty. Defaults to 1.1.
        do_sample (bool, optional): Whether to use sampling. Defaults to True.
        eos_token_ids (Optional[List[int]]): List of end-of-sequence token ids.

    Yields:
        str: The generated text, streamed token by token.
    """

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    streamer = TextStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})

    # Add eos_token_ids to the generate parameters.
    generation_kwargs = dict(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        streamer=streamer,
    )
    if eos_token_ids is not None:
        generation_kwargs["eos_token_ids"] = eos_token_ids

    # Run the generation in a separate thread to not block the main loop
    _ = model.generate(**generation_kwargs)  # No need to store the output, streamer handles printing

def main():
    model_name = "meta-llama/Llama-3.1-405B" 

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set.  Please set it and try again.")
        print("You can set it like this: export HF_TOKEN='YOUR_TOKEN'")
        print("or in Windows: set HF_TOKEN=YOUR_TOKEN")
        return

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name, use_auth_token=hf_token)

    if model is None or tokenizer is None:
        print("Failed to load the model.  Please check your model name and ensure you have the necessary files/permissions.")
        return

    # Check if the model was loaded successfully
    if model is not None:
        while True:
            prompt = input("Enter your prompt (or type 'exit' to quit): ")
            if prompt.lower() == "exit":
                break

            # Generate and stream the output
            generate_stream(model, tokenizer, prompt)
            print("\n")  # Add a newline for better readability


if __name__ == "__main__":  
    main()
