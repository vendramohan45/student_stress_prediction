from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import joblib
import json
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

MODEL_PATH = os.path.join("models", "best_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")
USER_DATA_PATH = os.path.join("data", "users.json")
HISTORY_DATA_PATH = os.path.join("data", "history.json")

# Load trained model & metrics
model = joblib.load(MODEL_PATH)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
else:
    metrics = {}

def load_users():
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "r") as f:
            return json.load(f)
    else:
        # Create default admin user if no users exist
        default_user = {
            "admin": {
                "password": generate_password_hash("adminpass"),
                "role": "admin"
            }
        }
        save_users(default_user)
        return default_user

def save_users(users):
    with open(USER_DATA_PATH, "w") as f:
        json.dump(users, f, indent=4)

def load_history():
    if os.path.exists(HISTORY_DATA_PATH):
        with open(HISTORY_DATA_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_history(history):
    with open(HISTORY_DATA_PATH, "w") as f:
        json.dump(history, f, indent=4)

users = load_users()
history_data = load_history()

# Define UI feature sliders (must match training features order)
FEATURES = [
    {
        "name": "study_hours_per_day",
        "label": "Study Hours per Day",
        "min": 0,
        "max": 10,
        "step": 0.5,
        "default": 4,
    },
    {
        "name": "assignments_per_week",
        "label": "Assignments per Week",
        "min": 0,
        "max": 10,
        "step": 1,
        "default": 4,
    },
    {
        "name": "exam_prep_days",
        "label": "Exam Preparation Days (per exam)",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 5,
    },
    {
        "name": "academic_pressure",
        "label": "Academic Pressure (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "anxiety_level",
        "label": "Anxiety Level (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 6,
    },
    {
        "name": "motivation_level",
        "label": "Motivation Level (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 5,
    },
    {
        "name": "sleep_hours",
        "label": "Sleep Hours per Night",
        "min": 3,
        "max": 10,
        "step": 0.5,
        "default": 7,
    },
    {
        "name": "exercise_days_per_week",
        "label": "Exercise Days per Week",
        "min": 0,
        "max": 7,
        "step": 1,
        "default": 2,
    },
    {
        "name": "screen_time_hours",
        "label": "Screen Time (Hours per Day)",
        "min": 0,
        "max": 12,
        "step": 0.5,
        "default": 6,
    },
    {
        "name": "family_support",
        "label": "Family Support (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 7,
    },
    {
        "name": "peer_support",
        "label": "Peer Support (1–10)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 6,
    },
    {
        "name": "commute_time_minutes",
        "label": "Commute Time (Minutes per Day)",
        "min": 0,
        "max": 120,
        "step": 5,
        "default": 30,
    },
]

def get_suggestion(stress_value: float) -> str:
    if stress_value < 30:
        return "Low stress. Keep maintaining your healthy habits 👍"
    elif stress_value < 60:
        return "Moderate stress. Try improving sleep, exercise and time management 🙂"
    elif stress_value < 80:
        return "High stress. Consider relaxation, talking to friends/family, or counseling ⚠️"
    else:
        return "Very high stress. Please seek support from a counselor or mentor immediately 🚨"

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    prediction = None
    suggestion = None
    input_values = {feat["name"]: feat["default"] for feat in FEATURES}

    if request.method == "POST":
        values_for_model = []
        for feat in FEATURES:
            raw_val = request.form.get(feat["name"], feat["default"])
            try:
                val = float(raw_val)
            except ValueError:
                val = float(feat["default"])
            input_values[feat["name"]] = val
            values_for_model.append(val)

        X = np.array(values_for_model).reshape(1, -1)
        stress_pred = float(model.predict(X)[0])
        stress_pred = max(0.0, min(100.0, stress_pred))  # clamp 0–100
        prediction = round(stress_pred, 2)
        suggestion = get_suggestion(prediction)

        # Save prediction to history
        user = session["username"]
        if user not in history_data:
            history_data[user] = []
        history_data[user].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stress_level": prediction
        })
        save_history(history_data)

    return render_template(
        "index.html",
        features=FEATURES,
        metrics=metrics,
        prediction=prediction,
        suggestion=suggestion,
        inputs=input_values,
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and check_password_hash(users[username]["password"], password):
            session["username"] = username
            session["role"] = users[username].get("role", "user")
            return redirect(url_for("index"))
        else:
            error = "Invalid username or password"
            return render_template("login.html", error=error)

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if username == "" or password == "" or confirm_password == "":
            error = "All fields are required."
            return render_template("register.html", error=error)

        if username in users:
            error = "Username already exists."
            return render_template("register.html", error=error)

        if password != confirm_password:
            error = "Passwords do not match."
            return render_template("register.html", error=error)

        hashed_password = generate_password_hash(password)
        users[username] = {"password": hashed_password, "role": "user"}
        save_users(users)
        flash("Registration successful! Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    user = session["username"]
    user_history = history_data.get(user, [])
    return render_template("history.html", user=user, history=user_history)

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    total_users = len(users)
    admin_count = sum(1 for u in users.values() if u.get("role") == "admin")
    recent_predictions = []
    for user, preds in history_data.items():
        for pred in preds[-5:]:  # last 5 predictions per user
            recent_predictions.append({"user": user, "date": pred["date"], "stress_level": pred["stress_level"]})
    recent_predictions.sort(key=lambda x: x["date"], reverse=True)
    recent_predictions = recent_predictions[:10]  # top 10 recent
    return render_template("dashboard.html", total_users=total_users, admin_count=admin_count, recent_predictions=recent_predictions, metrics=metrics)

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session or session.get("role") != "admin":
            flash("Admin access required")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user")

        if action == "add":
            if username in users:
                flash("User already exists.")
            else:
                if password == "":
                    flash("Password is required for new user.")
                else:
                    hashed_password = generate_password_hash(password)
                    users[username] = {"password": hashed_password, "role": role}
                    save_users(users)
                    flash(f"User {username} added successfully.")
        elif action == "edit":
            if username not in users:
                flash("User not found.")
            else:
                if password != "":
                    users[username]["password"] = generate_password_hash(password)
                users[username]["role"] = role
                save_users(users)
                flash(f"User {username} updated successfully.")
        elif action == "delete":
            if username in users:
                if username == session.get("username"):
                    flash("You cannot delete the currently logged-in admin user.")
                else:
                    users.pop(username)
                    save_users(users)
                    flash(f"User {username} deleted successfully.")
            else:
                flash("User not found.")

    all_users = [{"username": u, "role": users[u].get("role", "user")} for u in users]
    return render_template("admin.html", users=all_users)

if __name__ == "__main__":
    app.run(debug=True)
