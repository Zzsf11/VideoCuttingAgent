import argparse
import os
import sys

# Add the project root to sys.path to allow imports from vca
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vca.audio.interactive.interface import create_gradio_interface

def main():
    parser = argparse.ArgumentParser(description="Audio Segmentation Interactive Tool")
    parser.add_argument("--audio", type=str, default=None, help="Path to default audio file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()

    # Note: detection_method is now a Gradio interface option, not a command-line argument
    demo = create_gradio_interface(
        default_audio_path=args.audio,
        default_detection_method="downbeat"
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
