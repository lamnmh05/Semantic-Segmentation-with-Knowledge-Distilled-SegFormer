import sys
from src.train import main as train_main

def main():
    # Allow running on Kaggle by passing default config if no args provided
    if len(sys.argv) == 1:
        print("No arguments provided. Please run with --config path/to/config.yml")
        print("Example: python main.py --config configs/AttnFD_ADE20k.yml --data_path /kaggle/input/dataset --epochs 50")
    
    train_main()

if __name__ == "__main__":
    main()
