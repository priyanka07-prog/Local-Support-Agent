from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalGenerator:
    def __init__(self):
        print("Loading local language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

        self.model.eval()

        print("Model loaded successfully.")

    def generate(
        self,
        question: str,
        documents: list[dict],
    ) -> str:

        context_parts = []

        for document in documents:
            context_parts.append(
                f"Source: {document['source']}\n"
                f"{document['content']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are an OrbitDesk support assistant.

Answer the user's question using ONLY the provided knowledge-base
context.

If the context does not contain enough information to answer,
say that you do not have enough information.

User question:
{question}

Knowledge-base context:
{context}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()