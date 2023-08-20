from flask import Flask, render_template, request, redirect, url_for, flash
import os
import reference_extraction.reference_extractor_ml as ref_ext
import component_identification.component_identifier as comp_ident
import feedback_grading.grading as g

app = Flask(__name__, template_folder='templates')
app.secret_key = 'supersecretkey'

# Ensure the instance folder exists
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    return """
    <!doctype html>
    <head>
    <title>Upload new File</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <h1>Upload new File</h1>
    <form method=post action=/upload enctype=multipart/form-data id=uploadForm>
      <input type=file name=file>
      <input type=submit value=Upload id="processButton">
    </form>
    <div id="loadingIcon" style="display: none;">
        <img src="static/images/loading.gif" alt="Loading...">
    </div>
    <script>
        $('#processButton').click(function(event) {
            event.preventDefault();
            $(this).prop("disabled", true);  // Disable the button
            $('#loadingIcon').show();        // Show the loading icon
            $("#uploadForm").submit();

            
        });
    </script>
    """


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "please upload a file"

        file = request.files['file']

        # if user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "please select a file"

        if file:
            # Save the file to the UPLOAD_FOLDER
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filename)
            file.save(filename)
            flash('File successfully uploaded')
            ref_ext.extract_ref(filename)
            components = comp_ident.get_components()
            print(components)
            grades, fb = g.grade()
    return render_template('success.html', grades=grades, fb=fb)



@app.route('/test', methods=['GET'])
def bar():
    grades = [12.43, 43.23, 43.2]
    fb = ['for reference 11:', "hello1", "hello2", '\n', 'for reference 12:', 'fejfm', 'grdg', '\n']
    return render_template('success.html', grades=grades, fb=fb)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
