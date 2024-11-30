import torch
import torch.nn as nn
from data_generator import get_data_loaders
from model import DivisionModel
from torch.optim import Adam
import matplotlib.pyplot as plt
import sys
import select
import os
import argparse

def check_for_quit():
    # Windows doesn't support select on stdin
    if os.name == 'nt':
        import msvcrt
        if msvcrt.kbhit():
            if msvcrt.getch().decode().lower() == 'q':
                return True
    else:
        # Unix-like systems
        if select.select([sys.stdin], [], [], 0.0)[0]:
            if sys.stdin.read(1).lower() == 'q':
                return True
    return False

def train_model(lr=0.001, reset=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'division_model.pth'
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Initialize model, loss, and optimizer
    model = DivisionModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    current_epoch = 0
    
    # Load previous model if it exists and --reset wasn't specified
    if os.path.exists(save_path) and not reset:
        print(f"Loading existing model from {save_path}")
        checkpoint = torch.load(save_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        current_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {current_epoch}")
    else:
        print("Starting fresh training")
    
    print("Training continuously. Press 'q' and Enter to quit...")
    
    try:
        while True:  # Train continuously until interrupted
            current_epoch += 1
            model.train()
            train_loss = 0
            example_input = None
            example_output = None
            example_target = None
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                
                # Store first batch's first example for logging
                if example_input is None:
                    example_input = X[0].cpu().detach()
                    example_target = y[0].cpu().detach()
                
                # Forward pass
                outputs = model(X)
                
                # Store example output
                if example_output is None:
                    example_output = outputs[0].cpu().detach()
                
                loss = criterion(outputs, y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    val_loss += criterion(outputs, y).item()
            
            # Record losses
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            # Print status after every epoch with example calculation
            print(
                f'Epoch [{current_epoch}], '
                f'Train Loss: {train_losses[-1]:.6f}, '
                f'Val Loss: {val_losses[-1]:.6f}, '
                f'Example: {example_input[0]:.0f} / {example_input[1]:.0f} = {example_output.item():.3f} '
                f'(true: {example_target.item():.3f})'
            )
            
            if check_for_quit():
                print("\nTraining interrupted by user. Saving model...")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    
    finally:
        # Save the model
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, save_path)
        print(f"Model saved to {save_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.show()
        
        # Final example prediction
        model.eval()
        with torch.no_grad():
            example_X = torch.tensor([[4.0, 2.0]], device=device)
            example_y = model(example_X)
            print("\nFinal model test:")
            print(f"4 / 2 = {example_y.item():.3f} (expected: 2.000)")
        
        print("\nExiting program...")
        sys.exit(0)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train division model')
    parser.add_argument('--reset', action='store_true', help='Start fresh training instead of loading existing model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    model = train_model(lr=args.lr, reset=args.reset)