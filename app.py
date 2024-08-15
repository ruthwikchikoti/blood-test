import os
import traceback
from flask import Flask, render_template, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Define your agents
blood_test_analyzer = Agent(
    role='Blood Test Analyzer',
    goal='Analyze blood test reports and summarize findings',
    backstory="You're an expert in interpreting blood test results. You can identify abnormal values and their potential implications.",
    allow_delegation=False,
    llm=llm
)

article_searcher = Agent(
    role='Article Searcher',
    goal='Find relevant health articles based on blood test results',
    backstory="You're a skilled researcher with extensive knowledge of medical literature. You can find reputable sources and summarize their content.",
    allow_delegation=False,
    llm=llm
)

recommendation_maker = Agent(
    role='Recommendation Maker',
    goal='Provide health recommendations based on blood test results and articles',
    backstory="You're a health advisor with expertise in creating personalized health plans. You can interpret blood test results and research findings to provide actionable recommendations.",
    allow_delegation=False,
    llm=llm
)

def analyze_blood_test(report_content):
    try:
        # Define tasks
        task1 = Task(
            description="Analyze the blood test report, identify abnormal values, and summarize key findings. Explain the potential implications of these findings.",
            agent=blood_test_analyzer,
            expected_output="A detailed summary of key findings from the blood test report, including abnormal values and their potential implications."
        )

        task2 = Task(
            description="Based on the blood test analysis, search for relevant, reputable health articles. Focus on articles that address the specific health issues identified in the blood test.",
            agent=article_searcher,
            expected_output="A list of 3-5 relevant health articles with brief descriptions and URLs."
        )

        task3 = Task(
            description="Create personalized health recommendations based on the blood test results and the information from the found articles. Provide specific, actionable advice.",
            agent=recommendation_maker,
            expected_output="A set of 5-7 personalized health recommendations based on the blood test results and found articles, with explanations for each recommendation."
        )

        # Create crew
        crew = Crew(
            agents=[blood_test_analyzer, article_searcher, recommendation_maker],
            tasks=[task1, task2, task3],
            verbose=True
        )

        result = crew.kickoff()

        # Structure the result
        structured_result = {
            "blood_test_analysis": result.tasks[0].output,
            "relevant_articles": result.tasks[1].output,
            "health_recommendations": result.tasks[2].output
        }

        return structured_result
    except Exception as e:
        app.logger.error(f"An error occurred during analysis: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {"error": f"An error occurred during analysis: {str(e)}"}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            app.logger.info("POST request received")
            if 'file' not in request.files:
                app.logger.error("No file part in the request")
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            app.logger.info(f"File received: {file.filename}")
            
            if file.filename == '':
                app.logger.error("No selected file")
                return jsonify({'error': 'No selected file'}), 400
            
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                app.logger.info(f"File saved to {filepath}")
                
                # Extract text from PDF
                with open(filepath, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                app.logger.info("Text extracted from PDF")
                app.logger.info(f"Extracted text: {text[:500]}...")  # Log first 500 characters
                
                # Analyze the extracted text
                result = analyze_blood_test(text)
                app.logger.info(f"Analysis result: {result}")
                if isinstance(result, dict) and "error" in result:
                    app.logger.error(f"Error in analysis: {result['error']}")
                    return jsonify({"error": result["error"]}), 500
                return jsonify(result)
            else:
                app.logger.error("Invalid file type")
                return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': 'An internal server error occurred. Please try again later.'}), 500
        finally:
            # Clean up the uploaded file
            if 'filepath' in locals():
                os.remove(filepath)
                app.logger.info(f"Removed temporary file: {filepath}")
    return make_response(render_template('index.html'))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An unhandled exception occurred: {str(e)}")
    app.logger.error(traceback.format_exc())
    return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Changed default to 5001
    app.run(debug=True, host='0.0.0.0', port=port)