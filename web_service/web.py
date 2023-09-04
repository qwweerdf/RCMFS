from flask import Flask, render_template, request, flash
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

"""
a simple web application to provide an interface for users
"""


# main page
@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')


# upload POST request
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "please upload a file"

        # get parameters
        file = request.files['file']
        ref_ext_model = request.form.get('RefExtModel')
        comp_ident_model = request.form.get('CompIdentModel')
        if file is None or ref_ext_model is None or comp_ident_model is None:
            return "please pass all parameters"
        api = request.form.get('api')

        extension = os.path.splitext(file.filename)[1][1:]

        # if user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "please select a file"
        if extension != 'doc' and extension != 'docx' and extension != 'bib':
            return "file type not supported! supported document: doc/docx/bib"
        if file:
            # Save the file to the UPLOAD_FOLDER
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(filename)
            file.save(filename)
            flash('File successfully uploaded')



            # START PIPELINE
            if extension == 'docx' or extension == 'doc':
                refs = ref_ext.extract_ref(filename, model_type=ref_ext_model)
                if len(refs) == 0:
                    return "No reference extracted!"
            components = comp_ident.get_components(model_type=comp_ident_model, ner=True, ftype=extension)
            print(components)
            grades, fb, summ = g.grade()
            # END PIPELINE


            extracted_references = os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/extracted_references.txt'
            ref_compare = os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt'
            uploaded_file = os.getcwd() + '/' + 'instance/uploads/' + file.filename

            # delete temp files after completion
            if os.path.exists(extracted_references):
                os.remove(extracted_references)
                print(f"'{extracted_references}' has been deleted!")
            else:
                print(f"'{extracted_references}' does not exist!")
            if os.path.exists(ref_compare):
                os.remove(ref_compare)
                print(f"'{ref_compare}' has been deleted!")
            else:
                print(f"'{ref_compare}' does not exist!")
            if os.path.exists(uploaded_file):
                os.remove(uploaded_file)
                print(f"'{uploaded_file}' has been deleted!")
            else:
                print(f"'{uploaded_file}' does not exist!")
        # if using api, then a json format is returned
        if api == 'json':
            return {
                "file": file.filename,
                "grades": grades,
                "feedback": fb,
                "summary": summ
            }
        return render_template('success.html', grades=grades, fb=fb, summ=summ)
    else:
        return "HTTP Method Not Supported!"


# manually delete files
@app.route('/delete', methods=['GET'])
def delete():
    extracted_references = os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/extracted_references.txt'
    ref_compare = os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt'

    # delete temp files after completion
    if os.path.exists(extracted_references):
        os.remove(extracted_references)
        print(f"'{extracted_references}' has been deleted!")
    else:
        print(f"'{extracted_references}' does not exist!")
    if os.path.exists(ref_compare):
        os.remove(ref_compare)
        print(f"'{ref_compare}' has been deleted!")
    else:
        print(f"'{ref_compare}' does not exist!")
    return "done"


if __name__ == '__main__':
    app.run(port=9999, debug=True)
