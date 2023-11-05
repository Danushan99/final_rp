import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from flask import Flask, render_template, request
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


app = Flask(__name__)


# preprocess and OCR
def ocr_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    text = pytesseract.image_to_string(image)
    return text

# Define a dictionary for manual replacements


# correct text
# Modify the correct_text function
def correct_text(text):
    from language_tool_python import LanguageTool
    tool = LanguageTool('en-US')
    corrected_text = tool.correct(text)

    # Apply manual replacements
    for incorrect_word, correct_word in manual_replacements.items():
        corrected_text = corrected_text.replace(incorrect_word, correct_word)

    return corrected_text



# route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file provided."

        uploaded_image = request.files['image']

        if uploaded_image.filename == '':
            return "No selected image file."

        if uploaded_image:
            image_path = "uploaded_image.png"
            uploaded_image.save(image_path)
            ocr_result = ocr_image(image_path)
            corrected_result = correct_text(ocr_result)
            return render_template('index.html', ocr_result=ocr_result, corrected_result=corrected_result)

    return render_template('index.html', ocr_result=None, corrected_result=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3998, debug=True)