import torch
from model import DivisionModel
from data_generator import DivisionDataset
import numpy as np
import argparse

def evaluate_model(model_path='division_model.pth', numerator=None, denominator=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DivisionModel()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # If specific numbers were provided, calculate just those
    if numerator is not None and denominator is not None:
        with torch.no_grad():
            X = torch.tensor([[float(numerator), float(denominator)]], device=device)
            y_pred = model(X)
            
            # Handle division by zero for true result display
            if denominator == 0:
                true_result = "infinity"
            else:
                true_result = f"{(numerator / denominator):.3f}"
            
            print(f"\nCalculation:")
            print(f"{numerator} ÷ {denominator} = {y_pred.item():.3f} (True: {true_result})")
            return
    
    # Otherwise run the full evaluation
    test_dataset = DivisionDataset(1000)
    tolerance = 0.01
    correct = 0
    examples = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            X, y_true = test_dataset[i]
            X = X.to(device)
            
            # Make prediction
            y_pred = model(X)
            
            # Check accuracy within tolerance
            if abs(y_pred.item() - y_true.item()) < tolerance:
                correct += 1
            
            # Store some examples
            if i < 5:
                examples.append((X[0].item(), X[1].item(), y_pred.item(), y_true.item()))
    
    accuracy = correct / len(test_dataset)
    print(f"Model accuracy (within {tolerance} tolerance): {accuracy:.2%}")
    
    # Show example predictions
    print("\nExample predictions:")
    for num, den, pred, true in examples:
        print(f"{num:.0f} ÷ {den:.0f} = {pred:.2f} (True: {true:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate division model')
    parser.add_argument('numerator', nargs='?', type=float, help='Numerator for division')
    parser.add_argument('denominator', nargs='?', type=float, help='Denominator for division')
    
    args = parser.parse_args()
    
    if bool(args.numerator is not None) != bool(args.denominator is not None):
        parser.error("Both numerator and denominator must be provided together")
    
    evaluate_model(numerator=args.numerator, denominator=args.denominator)