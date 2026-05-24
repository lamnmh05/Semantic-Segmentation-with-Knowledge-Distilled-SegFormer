import sys
from src.train import main as train_main

def main():
    if len(sys.argv) == 1:
        print("No arguments provided. Please run with --config path/to/config.yml")
        print("Example: python main.py --config configs/AttnFD_ADE20k.yml")
    
    train_main()

if __name__ == "__main__":
    main()
