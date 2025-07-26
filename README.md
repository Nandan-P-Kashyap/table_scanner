Developed a Dockerized OCR pipeline to convert scanned hardcopy bills into structured product metadata.
Parsed OCR output into YAML, capturing product names, quantities, prices, and delivery annotations.
Currently integrating LLM to detect delivery status (✓, ✗, Yes, No) from noisy OCR output with improved accuracy.
Outputs complaint-related product details in JSON format for use in customer support systems.
Modular, extensible design with centralized logging and environment-isolated Docker setup.
Tools and technologies:Python, Tesseract OCR, OpenCV, Docker, YAML, JSON, Gemini API, Logging (YAML config)
