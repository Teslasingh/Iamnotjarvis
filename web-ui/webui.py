from pathlib import Path

from dotenv import load_dotenv

_WEBUI_ROOT = Path(__file__).resolve().parent
load_dotenv(_WEBUI_ROOT / ".env")
load_dotenv()

import argparse

from src.webui.interface import theme_map, create_ui


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    from src.browser.ocr_service import find_tesseract_executable, is_ocr_available

    if not is_ocr_available():
        hint = find_tesseract_executable()
        if not hint:
            print(
                "\n[OCR] Tesseract binary not found. pytesseract alone is not enough.\n"
                "  Install: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Then add to web-ui/.env:\n"
                "  TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
            )

    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()
