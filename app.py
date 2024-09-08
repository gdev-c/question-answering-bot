from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json
from model import main

app = Flask(__name__)

@app.route('/')
def upload_files():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        json_file = request.files['json_file']
        
        if pdf_file and json_file:
            try:
                pdf_content = pdf_file.read()
                json_content = json_file.read()
                
                answers = main(pdf_content, json_content)
                
                formatted_answers = json.dumps(answers, indent=2)

                return render_template('index.html', answers=formatted_answers)
            except Exception as e:
                return f"An error occurred: {str(e)}"
    
    return "Invalid request"

if __name__ == "__main__":
    app.run(debug=True)