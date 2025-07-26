import base64
import json
import logging
import logging.config
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from myocr.pipelines.common_ocr_pipeline import CommonOCRPipeline
from myocr.pipelines.structured_output_pipeline import StructuredOutputOCRPipeline
from myocr.utils import extract_image_type

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Enable CORS for endpoints
CORS(app, resources={r"/ocr": {"origins": "*"}, r"/ocr-json": {"origins": "*"}})

# Load logging configuration
def setup_logging(config_path: str = "logging_config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            logger.debug("Logging configured successfully.")
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Failed to load logging config: {e}")

setup_logging()

# Structured data models
class InvoiceItem(BaseModel):
    name: str
    price: float
    number: str
    tax: str

class InvoiceModel(BaseModel):
    invoiceNumber: str
    invoiceDate: str
    invoiceItems: list[InvoiceItem]
    totalAmount: float

# Initialize pipelines
global common_ocr_pipeline, structured_pipeline
common_ocr_pipeline = CommonOCRPipeline("cuda:0")
structured_pipeline = StructuredOutputOCRPipeline("cuda:0", InvoiceModel)

@app.route("/ping")
def ping():
    return "pong"

# Core OCR handler
def _do_ocr(pipeline, structured: bool = False):
    logger.debug("_do_ocr called (structured=%s)", structured)
    try:
        # 1. Read and validate JSON payload
        data = request.get_json(force=True)
        logger.debug("Request JSON: %s", data)

        image_data = data.get("image")
        if not image_data:
            logger.warning("No 'image' field in request.")
            return jsonify({"error": "No image data provided"}), 400

        # 2. Extract image type and base64 string
        image_type, b64 = extract_image_type(image_data)
        logger.debug("Extracted image type: %s", image_type)
        if not image_type or not b64:
            logger.warning("Invalid base64 image data.")
            return jsonify({"error": "Invalid base64 image data"}), 400

        # 3. Decode base64
        try:
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            logger.error("Base64 decoding failed: %s", e)
            return jsonify({"error": "Malformed base64 data"}), 400

        # 4. Run OCR pipeline
        raw_output = pipeline(image_bytes)
        logger.debug("Raw pipeline output: %s", raw_output)

        # 5. If structured, validate and return full model
        if structured:
            try:
                if isinstance(raw_output, (str, bytes)):
                    raw_json = json.loads(raw_output)
                elif hasattr(raw_output, "to_dict"):
                    raw_json = raw_output.to_dict()
                else:
                    raw_json = raw_output

                model = InvoiceModel(**raw_json)
                result = model.dict()
                logger.debug("Structured output validated successfully.")
                return jsonify(result)
            except (ValidationError, json.JSONDecodeError) as e:
                logger.error("Structured parsing failed: %s", e)
                # fallback to plain text

        # 6a. If it's an object with .regions (e.g. OCRResult), extract each region.text
    # ——— New 6a: group OCRResult.regions into table rows ———
        if hasattr(raw_output, "regions"):
            # 1. Build a list of (region, centroid_x, centroid_y)
            regs = []
            for r in raw_output.regions:
                pts = r.bounding_shape.points
                avg_x = sum(p.x for p in pts) / len(pts)
                avg_y = sum(p.y for p in pts) / len(pts)
                regs.append((r, avg_x, avg_y))

            # 2. Sort by y, then x
            regs.sort(key=lambda t: (t[2], t[1]))

            # 3. Threshold to decide new rows (20px here)
            rows: list[list[str]] = []
            current_row_y = None
            current_cells: list[tuple[float,str]] = []

            for r, x, y in regs:
                text = r.text.strip()
                if not text:
                    continue
                if current_row_y is None:
                    # first cell
                    current_row_y = y
                # if gap to new region is big, flush the old row
                if abs(y - current_row_y) > 20:
                    # sort old row by x and store just the texts
                    cells = [cell for _, cell in sorted(current_cells, key=lambda c: c[0])]
                    rows.append(cells)
                    current_cells = []
                    current_row_y = y
                current_cells.append((x, text))
            # flush last row
            if current_cells:
                cells = [cell for _, cell in sorted(current_cells, key=lambda c: c[0])]
                rows.append(cells)

            # 4. Join each row’s cells into a single string
            lines = ["\t".join(row) for row in rows]
            clean_text = "\n".join(lines)
            logger.debug("Table‐structured text:\n%s", clean_text)
            return jsonify({"text": clean_text})

        # 6b. Other fallbacks: dicts, lists, strings, etc.
        text_lines = []
        if isinstance(raw_output, str):
            text_lines = [raw_output]
        elif isinstance(raw_output, dict) and "regions" in raw_output:
            text_lines = [r.get("text", "") for r in raw_output["regions"] if r.get("text")]
        elif isinstance(raw_output, (list, tuple)):
            for item in raw_output:
                if isinstance(item, (list, tuple)) and item:
                    text_lines.append(str(item[0]))
        else:
            text_lines = [str(raw_output)]

        clean_text = "\n".join([t.strip() for t in text_lines if t.strip()])
        logger.debug("Extracted text (fallback): %s", clean_text)
        return jsonify({"text": clean_text})

    except Exception as e:
        logger.error("Unhandled OCR error: %s", exc_info=True)
        return jsonify({"error": "Internal OCR error"}), 500
    
# Your extraction logic
def extract_missing_products(ocr_text):
    lines = ocr_text.split('\n')
    missing_products = []

    for line in lines:
        # if 'no' in line.lower() or 'x' in line.lower():
        #     parts = line.split('\t')
        #     if len(parts) > 0:
        #         product_name = parts[0].strip()
        #         missing_products.append(product_name)
        #     else:
        #         product_name = line.split(' ')[0]
        #         missing_products.append(product_name)
        if 'no' in line.lower() or 'x' in line.lower():
            words = [word.strip() for word in line.split('\t') if word.strip()]
            missing_products.append(words)

    return missing_products

# Route for processing OCR text
@app.route('/process-ocr', methods=['POST'])
def process_ocr():
    data = request.get_json()
    ocr_text = data.get('ocr_text', '')

    if not ocr_text:
        return jsonify({'error': 'No OCR text provided'}), 400

    missing_items = extract_missing_products(ocr_text)
    return jsonify({'missing_items': missing_items}), 200

@app.route("/ocr", methods=["POST"])
def ocr():
    return _do_ocr(common_ocr_pipeline, structured=False)

@app.route("/ocr-json", methods=["POST"])
def ocr_json():
    template = request.json.get("template", "invoice")
    if template != "invoice":
        return jsonify({"error": f"Unsupported template: {template}"}), 400
    structured_pipeline.set_response_format(InvoiceModel)
    return _do_ocr(structured_pipeline, structured=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
