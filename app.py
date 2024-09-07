from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
from model import main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"

if not os.path.exists("./uploads"):
    os.makedirs("./uploads")

@app.route('/')
def upload_files():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    #print("post request triggered")
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        json_file = request.files['json_file']
        #print(type(request.files))
        if pdf_file and json_file:
            pdf_filename = secure_filename(pdf_file.filename)
            json_filename = secure_filename(json_file.filename)
            pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename))
            json_file.save(os.path.join(app.config['UPLOAD_FOLDER'], json_filename))
            
            pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            
            try:
                answers = main(pdf_file_path, json_file_path)
                
                formatted_answers = json.dumps(answers, indent=2)

                return render_template('index.html', answers=formatted_answers)
            except Exception as e:
                return f"An error occurred: {str(e)}"
    
    return "Invalid request"

if __name__ == "__main__":
    app.run(debug=True)
