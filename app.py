# Author: Prashant Mittal

from flask import Flask, render_template, request, redirect, session
import psycopg2
import os
import subprocess
from datetime import datetime, timezone
import secrets
import shutil

app = Flask(__name__)
app.secret_key = "your_secret_key"


# List to store queries and responses
queries = {}
queries[""] = []


def save_queries_to_file(username):
    with open(f"queries/{username}/queries.txt", "w") as file:
        for project_name, project_queries in queries.items():
            file.write(f"Project: {project_name}\n")
            for query_response in project_queries:
                query = query_response["query"]
                response = query_response["response"]
                file.write(f"Query: {query}\n")
                file.write(f"Response: {response}\n")
            file.write("\n")


def load_queries_from_file(username):
    project_name = ""
    query = ""
    response = ""
    if os.path.isfile(f"queries/{username}/queries.txt"):
        queries.clear()
        queries[""] = []
        with open(f"queries/{username}/queries.txt", "r") as file:
            is_query = False
            is_response = False
            for line in file:
                line = line.strip()
                if line.startswith("Project:"):
                    if response.startswith("("):
                        queries[project_name].append(
                            {"query": query, "response": response}
                        )
                    query = ""
                    response = ""
                    project_name = line[9:]
                    queries[project_name] = []
                    is_query = False
                    is_response = False
                elif (
                    line.startswith("Response:") or is_response
                ) and not line.startswith("Query:"):
                    is_response = True
                    if line.startswith("Response:"):
                        response += line[10:]
                    else:
                        response += line
                elif line.startswith("Query:") or is_query:
                    if response.startswith("("):
                        queries[project_name].append(
                            {"query": query, "response": response}
                        )
                    response = ""
                    query = ""
                    is_query = True
                    if line.startswith("Query:"):
                        query += line[7:]
                    else:
                        query += line
            if response.startswith("("):
                queries[project_name].append({"query": query, "response": response})
            print(queries)


def get_db_connection():
    # connection = psycopg2.connect(
    #     host='localhost',
    #     port='5432',
    #     dbname='knowledge_maker',
    #     user='postgres',
    #     password='hitr'
    # )
    connection = psycopg2.connect(
        host="localhost", port="5432", dbname="chatgpt", user="postgres", password=""
    )
    return connection


@app.before_request
def before_request():
    if request.endpoint == "login" and "user_id" in session:
        return redirect("/home")


@app.route("/delete_project", methods=["POST"])
def delete_project():
    if "user_id" in session:
        # Get the project_name from the request
        project_name = request.form["project_name"]
        if project_name == session["project_name"]:
            session["project_name"] = ""

        # Retrieve the username from the session
        username = session.get("username")

        queries.pop(project_name)
        save_queries_to_file(username)

        # Specify the folder path where the projects are saved
        folder_path = "source_documents/" + username

        # Build the full path of the directory to delete
        project_path = os.path.join(folder_path, project_name)

        try:
            # Check if the directory exists
            if os.path.isdir(project_path):
                # Delete the directory and its contents
                shutil.rmtree(project_path)
        except Exception as e:
            # Handle any errors that occur during directory deletion
            pass

    return redirect("/home")  # Redirect to the home page


@app.route("/delete", methods=["POST"])
def delete_file():
    if "user_id" in session:
        # Get the filename from the request
        filename = request.form["filename"]

        # Retrieve the username from the session
        username = session.get("username")
        project_name = session.get("project_name")

        # Specify the folder path where the uploaded files are saved
        folder_path = "source_documents/" + username + "/" + project_name

        # Build the full path of the file to delete
        file_path = os.path.join(folder_path, filename)

        try:
            # Check if the file exists
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
        except Exception as e:
            # Handle any errors that occur during file deletion
            pass

    return redirect("/home")  # Redirect to the home page


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT id, password FROM users WHERE username = %s", (username,)
        )
        user = cursor.fetchone()

        if user and user[1] == password:
            session["user_id"] = user[0]
            session["username"] = username
            session["start_time"] = datetime.now(timezone.utc)
            session["session_id"] = secrets.token_hex(16)  # Generate a session ID
            session["project_name"] = ""

            # Create the "templates/mitta" folder if it doesn't exist
            folder_path = os.path.join("source_documents", username)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path = os.path.join("queries", username)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Insert the session start time and username into the sessions table
            time_start = session["start_time"]
            time_start_str = time_start.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO sessions (session_id, username, time_start) VALUES (%s, %s, %s)",
                (session["session_id"], username, time_start_str),
            )
            connection.commit()
            return redirect("/home")
        else:
            error = "Invalid username or password"
            return render_template("login.html", error=error)

        cursor.close()
        connection.close()

    return render_template("login.html")


@app.route("/home")
def home():
    if "user_id" in session:
        # Retrieve the username from the session
        username = session.get("username")
        load_queries_from_file(username)
        project_name = session.get("project_name")
        # Specify the folder path where the uploaded files will be saved
        folder_path = "source_documents/" + username

        # Get the list of projects
        projects = os.listdir(folder_path)

        folder_path += "/" + project_name
        files = os.listdir(folder_path)
        file_names = []
        for file in files:
            if os.path.isfile(os.path.join(folder_path, file)):
                file_names.append(file)

        return render_template(
            "index.html",
            projects=projects,
            files=file_names,
            queries=queries[project_name],
            username=username,
            project_name=project_name,
        )
    else:
        return redirect("/login")


@app.route("/")
def index():
    if "user_id" in session:
        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/logout")
def logout():
    if "user_id" in session:
        connection = get_db_connection()
        cursor = connection.cursor()

        session_id = session.get("session_id")
        time_start = session.get("start_time")
        time_end = datetime.now(timezone.utc)

        time_end_str = time_end.strftime("%Y-%m-%d %H:%M:%S")
        duration = time_end - time_start

        # Update the session end time and duration in the sessions table
        cursor.execute(
            "UPDATE sessions SET time_end = %s, duration = %s WHERE session_id = %s",
            (time_end_str, duration, session_id),
        )
        connection.commit()

        cursor.close()
        connection.close()

    session.clear()
    return redirect("/login")


@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" in session:
        username = session.get("username")
        project_name = session.get("project_name")
        # Specify the folder path where the uploaded files will be saved
        folder_path = "source_documents/" + username

        # Check if the project folder exists, create it if it doesn't
        project_folder_path = os.path.join(folder_path, project_name)
        if not os.path.exists(project_folder_path):
            os.makedirs(project_folder_path)

        # Get the uploaded file(s)
        uploaded_files = request.files.getlist("file")
        timestamp = datetime.now(timezone.utc)

        # Save the uploaded file(s) to the project folder
        for file in uploaded_files:
            file.save(os.path.join(project_folder_path, file.filename))

            # Insert the document name, username, project name, and timestamp into the uploaded_documents table
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO uploaded_documents (document_name, username, project_name, timestamp) VALUES (%s, %s, %s, %s)",
                (file.filename, username, project_name, timestamp),
            )
            connection.commit()
            cursor.close()
            connection.close()

        # Run the first Python program on each uploaded file
        python_file_path = "ingest.py"
        process = subprocess.Popen(
            ["python3", python_file_path, username, project_name]
        )
        process.wait()

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/summary", methods=["POST"])
def summary():
    if "user_id" in session:
        username = session.get("username")
        project_name = session.get("project_name")
        file_path = request.form["filename"]

        # Run the second Python program and capture its output
        python_file_path = "summarizer.py"
        process = subprocess.Popen(
            [
                "python3",
                python_file_path,
                username,
                project_name,
                file_path,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()

        process.wait()

        # Extract the response from the output
        response_start_index = output.index("285")
        trimmed_output = output[response_start_index + 3 :].strip()

        print(trimmed_output)

        # Create a dictionary with query and response
        query_response = {"query": "Summary", "response": trimmed_output}

        # Add the query and response to the list
        queries[project_name].append(query_response)
        save_queries_to_file(username)

        return trimmed_output
    else:
        return redirect("/login")


@app.route("/query", methods=["POST"])
def query():
    if "user_id" in session:
        # Get the user query from the AJAX request
        user_query = request.form["query"]
        username = session.get("username")
        project_name = session.get("project_name")

        # Run the second Python program and capture its output
        python_file_path = "privateGPT.py"
        process = subprocess.Popen(
            [
                "python3",
                python_file_path,
                "--user",
                username,
                "--project",
                project_name,
                "--query",
                user_query,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate()

        process.wait()

        # Extract the response from the output
        response_start_index = output.index("> Answer")
        trimmed_output = output[response_start_index + 8 :].strip()

        print(trimmed_output)

        # Create a dictionary with query and response
        query_response = {"query": user_query, "response": trimmed_output}

        # Add the query and response to the list
        queries[project_name].append(query_response)
        save_queries_to_file(username)

        return trimmed_output
    else:
        return redirect("/login")


@app.route("/create_project", methods=["POST"])
def create_project():
    if "user_id" in session:
        username = session.get("username")
        project_name = request.form["project_name"]
        session["project_name"] = project_name
        queries[project_name] = []
        save_queries_to_file(username)

        # Specify the folder path where the uploaded files will be saved
        folder_path = "source_documents/" + username

        # Check if the project folder exists, create it if it doesn't
        project_folder_path = os.path.join(folder_path, project_name)
        if not os.path.exists(project_folder_path):
            os.makedirs(project_folder_path)

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/select_project", methods=["POST"])
def select_project():
    if "user_id" in session:
        project_name = request.form["project_name"]
        session["project_name"] = project_name

        return redirect("/home")
    else:
        return redirect("/login")


@app.route("/reset", methods=["POST"])
def reset_queries():
    if "user_id" in session:
        username = session["username"]
        project_name = session["project_name"]
        queries[project_name].clear()
        save_queries_to_file(username)
        return redirect("/home")
    else:
        return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True)
