# libraries
import random
import numpy as np
import pickle
import json
import bcrypt
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL

# Initialize Flask app
app = Flask(__name__)
app.secret_key = b'8\xa7n\xc1z&E\x89\xeeL\xcbW\x06`.7\x0f;\xc0s\x0e\x95|\xfb'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '3435'
app.config['MYSQL_DB'] = 'mydatabase'

mysql = MySQL(app)

# Load the model and other necessary data
model = load_model("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\chatbot_model.h5")
data_file = open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\intents.json").read()
intents = json.loads(data_file)
words = pickle.load(open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\words.pkl", "rb"))
classes = pickle.load(open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\classes.pkl", "rb"))
lemmatizer = WordNetLemmatizer()

# Routes
@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]

    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name) 
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email_or_username = request.form['login_email']
        password = request.form['login_password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email_or_username, email_or_username))
        user = cursor.fetchone()
        cursor.close()

        if user:
            # Verify password securely
            hashed_password = user[3].encode('utf-8')
            try:
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    return render_template('index.html')  # Redirect to the welcome route
                else:
                    # Incorrect password
                    flash('Invalid password. Please try again.')
                    return redirect(url_for('home'))
            except ValueError:
                # Handle bcrypt ValueError (log error, do not show to user)
                app.logger.error('Error: Invalid salt')
                flash('An error occurred. Please try again.', 'error')
                return redirect(url_for('home'))
        else:
            # User not found
            flash('User not found. Please register.')
            return redirect(url_for('home'))

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        email = request.form['register_email']
        username = request.form['register_username']
        password = request.form['register_password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # User already exists
            flash('User already exists.')
            return redirect(url_for('home'))
        else:
            # Hash password before storing
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            cursor.execute("INSERT INTO users (email, username, password) VALUES (%s, %s, %s)", (email, username, hashed_password))
            mysql.connection.commit()
            cursor.close()

            flash('Registered successfully!')  # Set success message
            return redirect(url_for('home'))  # Redirect after setting flash message

# Chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    result = "Sorry"
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(debug=True)
