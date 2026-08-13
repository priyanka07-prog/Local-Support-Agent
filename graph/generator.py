from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalGenerator:
    def __init__(self):
        print("Loading local language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

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
                f"[SOURCE: {document['source']}]\n"
                f"{document['content']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a careful OrbitDesk support assistant.

You must answer the user's question using ONLY the
knowledge-base context provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent permissions, steps, or product behavior.
3. Pay close attention to words such as:
   "only", "cannot", "not allowed", and "must".
4. If the context says only certain roles can perform
   an action, do not say that other roles can perform it.
5. Answer the user's exact question directly.
6. Keep the answer concise.
7. Do not add unsupported advice.
8. If the context does not contain enough information,
   say that you do not have enough information.

USER QUESTION:
{question}

KNOWLEDGE-BASE CONTEXT:
{context}

Check that your answer agrees with the context.

FINAL ANSWER:
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
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()