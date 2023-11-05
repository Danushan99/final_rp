import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

pytesseract.pytesseract.tesseract_cmd =  '/opt/homebrew/bin/tesseract'

# Load  image
image_path = 'f2.PNG'
image = Image.open(image_path)

# Preprocess
image = image.convert('L')  
image = image.filter(ImageFilter.MedianFilter()) 
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)  

# Perform OCR 
text = pytesseract.image_to_string(image)



import nltk
from language_tool_python import LanguageTool

tool = LanguageTool('en-US')

def correct_text(text):
    corrected_text = tool.correct(text)
    return corrected_text

input_text = text
corrected_text = correct_text(input_text)
print(corrected_text)