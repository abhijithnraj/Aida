import argparse
import logging
from pathlib import Path
from .core import Aida
from .config import AidaConfig
from .gui import main as gui_main

def main():
    parser = argparse.ArgumentParser(description="AIDA - AI Server Management Assistant")
    parser.add_argument("--core-model", help="Name of the LLM model to use for core functionality")
    parser.add_argument("--preprocessor-model", help="Name of the LLM model to use for preprocessing")
    parser.add_argument("--provider", help="Name of the LLM provider to use for both core and preprocessing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI interface")
    args = parser.parse_args()

    # If GUI mode is requested, launch it
    if args.gui:
        gui_main()
        return

    # Load config from file and update with CLI args
    config = AidaConfig.from_file(args.config)
    config.update_from_args(args)

    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print(f"Initializing AIDA with:")
    print(f"  Core model: {config.core_model}")
    print(f"  Preprocessor model: {config.preprocessor_model}")
    print(f"  Debug mode: {'enabled' if config.debug else 'disabled'}")
    
    aida = Aida(config=config)
    print("\nAIDA is ready! Type 'exit' to quit.")
    print("Type 'debug' to toggle debug mode.")
    print("Type 'config' to show current configuration.")
    
    debug_mode = config.debug
    
    while True:
        try:
            query = input("\nWhat can I help you with? > ").strip()
            if query.lower() == 'exit':
                break
            elif query.lower() == 'debug':
                debug_mode = not debug_mode
                logging.getLogger().setLevel(logging.DEBUG if debug_mode else logging.INFO)
                print(f"\nDebug mode: {'enabled' if debug_mode else 'disabled'}")
                continue
            elif query.lower() == 'config':
                print(f"\nCurrent configuration:")
                print(f"  Core model: {config.core_model}")
                print(f"  Preprocessor model: {config.preprocessor_model}")
                print(f"  Debug mode: {'enabled' if config.debug else 'disabled'}")
                continue
            elif not query:
                continue
            
            print("\nProcessing your request...")
            response = aida.process_query(query)
            print("\nAIDA:", response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
