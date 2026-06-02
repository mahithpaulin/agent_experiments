import numpy as np
from model import LinearModel
from trainer import Trainer
from evaluator import Evaluator
from utils import generate_data

def main():
    # Generate synthetic data
    X, y = generate_data(num_samples=100, noise=0.1)

    # Initialize model
    model = LinearModel(input_dim=X.shape[1])

    # Train model
    trainer = Trainer(model=model, learning_rate=0.01, epochs=1000)
    trainer.train(X, y)

    # Evaluate model
    evaluator = Evaluator(model=model)
    mse = evaluator.evaluate(X, y)

    print(f"Final Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()