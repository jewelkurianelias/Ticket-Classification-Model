import pandas as pd
import random
import os

def generate_ticket_data(num_samples: int = 500, output_path: str = "data/tickets.csv"):
    """Generates synthetic IT support tickets for classification."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    categories = ["bug", "billing", "feature request"]
    
    # Vocabulary to ensure the model can actually learn patterns
    vocab = {
        "bug": ["error", "crash", "broken", "won't load", "exception", "failed", "blank screen", "stuck"],
        "billing": ["invoice", "charge", "refund", "credit card", "payment", "subscription", "declined"],
        "feature request": ["add", "support", "integration", "dark mode", "export", "new button", "wish"]
    }
    
    data = []
    for _ in range(num_samples):
        cat = random.choice(categories)
        # Build a synthetic sentence
        words = random.sample(vocab[cat], k=random.randint(1, 3))
        filler_start = ["I have an issue with", "Please help with", "Why is there a", "Can you"]
        text = f"{random.choice(filler_start)} {' and '.join(words)}"
        data.append({"text": text, "category": cat})
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {num_samples} support tickets at {output_path}")

if __name__ == "__main__":
    generate_ticket_data()