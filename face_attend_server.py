# %%
from flask import Flask, request, jsonify  # Core Flask tools
import json  # JSON parsing
from flask_cors import CORS  # Enable cross-origin for React
import cv2  # Image decoding
import numpy as np  # Array ops
import pickle  # Load model
import sqlite3  # Local database
from datetime import datetime  # Time stamping
import logging  # Logging
from skimage.feature import hog  # HOG extraction
import os  # Directory checks
from sklearn.svm import SVC  # SVM refit
from sklearn.preprocessing import StandardScaler  # Scaling
from sklearn.model_selection import train_test_split  # Retrain split
from sklearn.metrics import accuracy_score  # Retrain eval

app = Flask(__name__)  # Init app
CORS(app, origins=["http://localhost:5173"])  # Unique: Specific origin for Vite

# Sub-action 1: Log setup
log_path = "activity_logs"  # Reuse directory
if not os.path.exists(log_path):
    os.makedirs(log_path)
server_log = f'{log_path}/server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=server_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sub-action 2: Model load
data_store = "ready_face_data"  # Data path
model_path = "svm_face_model.pkl"  # Model file
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_handle:  # Sub-action: Deserialize
        loaded_pack = pickle.load(model_handle)
        face_recognizer = loaded_pack['svm_model']  # Extract classifier
        feat_normalizer = loaded_pack['feature_scaler']  # Extract scaler
        label_key = loaded_pack['tag_dict']  # Extract map
    logging.info("Model loaded from file.")
else:
    logging.error("Model file missing - run Phase 2.")
    face_recognizer, feat_normalizer, label_key = None, None, {}

def get_hog_desc(gray_crop):  # Renamed, tweaked params
    """Sub-action: Generate HOG from grayscale crop."""
    desc_vector, _ = hog(gray_crop, orientations=8, pixels_per_cell=(10, 10),  # Unique params
                         cells_per_block=(2, 2), visualize=True, feature_vector=True)
    return desc_vector  # Return descriptor

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load cascade

def find_valid_face(input_frame):  # Renamed
    """Main action: Detect one face, convert/resize to gray.
    Sub-action: Convert to gray.
    Testable: Log detection count.
    """
    gray_input = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)  # Sub-action: To grayscale
    detected = face_detect.detectMultiScale(gray_input, 1.3, 5)  # Sub-action: Detect
    if len(detected) == 1:  # Sub-action: Ensure single face (anti-proxy)
        x_coord, y_coord, width, height = detected[0]
        face_crop = gray_input[y_coord:y_coord+height, x_coord:x_coord+width]  # Crop
        resized_crop = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_LINEAR)  # Resize
        unique_vals = np.unique(resized_crop)  # Sub-action: Grayscale check
        if len(unique_vals) <= 2:  # Basic gray validation
            logging.info("Valid single grayscale face detected.")  # Log success
            return resized_crop
    logging.warning(f"Invalid detection: {len(detected)} faces")  # Log issue
    return None  # No valid face

def get_db_connection():  # Unique function for DB
    """Sub-action: Establish SQLite connection."""
    return sqlite3.connect('attendance.db')

@app.route('/process_entry', methods=['POST'])  # Unique endpoint
def process_entry():
    """Main action: Handle attendance scan.
    Sub-action: Validate image file.
    Testable: Log confidence.
    """
    if 'face_capture' not in request.files:  # Sub-action: Check file
        logging.error("Missing face image in request.")
        return jsonify({'error': 'No image provided'}), 400

    file_handle = request.files['face_capture']  # Get file
    buffer_array = np.frombuffer(file_handle.read(), np.uint8)  # Sub-action: Buffer
    decoded_frame = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)  # Decode

    valid_crop = find_valid_face(decoded_frame)  # Sub-action: Detect face
    if valid_crop is None:
        return jsonify({'status': 'absent', 'message': 'No valid grayscale face'})

    crop_features = get_hog_desc(valid_crop)  # Sub-action: Extract HOG
    normalized_crop = feat_normalizer.transform([crop_features])  # Normalize
    prob_scores = face_recognizer.predict_proba(normalized_crop)[0]  # Sub-action: Predict probs
    top_idx = np.argmax(prob_scores)  # Get top class
    match_conf = prob_scores[top_idx] * 100  # Confidence calc
    logging.info(f"Entry scan confidence: {match_conf:.2f}%")  # Test log

    if match_conf < 95:  # Sub-action: Threshold check (SRS req)
        return jsonify({'status': 'absent', 'message': 'Confidence too low'})

    matched_tag = list(label_key.keys())[top_idx]  # Get tag
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM student_profiles WHERE full_name = ?', (matched_tag,))
    profile_result = cursor.fetchone()  # Sub-action: Fetch profile
    conn.close()

    if not profile_result:
        return jsonify({'status': 'unknown'})

    user_id = profile_result[0]  # Extract ID
    current_moment = datetime.now()  # Sub-action: Get time
    entry_stamp = current_moment.strftime("%H:%M:%S")  # Format time
    day_key = current_moment.strftime("%Y-%m-%d")  # Day for DB

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT present FROM daily_attendance WHERE date = ? AND id = ?', (day_key, user_id))
    existing_record = cursor.fetchone()  # Sub-action: Check duplicate
    if existing_record and existing_record[0]:
        conn.close()
        return jsonify({'status': 'already_marked'})

    cursor.execute('INSERT OR REPLACE INTO daily_attendance (date, id, present, in_time, out_time, duration) VALUES (?, ?, ?, ?, ?, ?)',
                   (day_key, user_id, 1, entry_stamp, None, 0))  # Sub-action: Save record
    conn.commit()
    conn.close()
    logging.info(f"Entry marked for {matched_tag} at {entry_stamp}")  # Log mark

    # Sub-action: Check for alerts (SRS req)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM daily_attendance WHERE id = ? AND present = 0', (user_id,))
    absent_count = cursor.fetchone()[0]
    conn.close()
    if absent_count >= 3:
        logging.warning(f"Alert: High absences for {matched_tag}")  # Alert log

    return jsonify({'status': 'present', 'name': matched_tag, 'id': user_id, 'in_time': entry_stamp, 'confidence': f"{match_conf:.2f}"})

@app.route('/handle_checkout', methods=['POST'])  # Unique name
def handle_checkout():
    """Main action: Log out time and duration.
    Sub-action: Validate JSON data.
    Testable: Log duration.
    """
    checkout_data = request.json  # Get JSON
    if not checkout_data or 'id' not in checkout_data:  # Sub-action: Validate
        logging.error("Invalid checkout data.")
        return jsonify({'error': 'Invalid data'}), 400

    user_id = checkout_data['id']  # Extract ID
    now_time = datetime.now()  # Get current time
    exit_stamp = now_time.strftime("%H:%M:%S")  # Format
    day_key = now_time.strftime("%Y-%m-%d")  # Day key

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT in_time FROM daily_attendance WHERE date = ? AND id = ?', (day_key, user_id))
    record_dict = cursor.fetchone()  # Fetch record

    if record_dict and not cursor.execute('SELECT out_time FROM daily_attendance WHERE date = ? AND id = ?', (day_key, user_id)).fetchone()[0]:  # Sub-action: Check open entry
        entry_time_obj = datetime.strptime(record_dict[0], "%H:%M:%S")  # Parse in
        exit_time_obj = datetime.strptime(exit_stamp, "%H:%M:%S")  # Parse out
        time_diff = int((exit_time_obj - entry_time_obj).total_seconds() / 60)  # Calc mins
        cursor.execute('UPDATE daily_attendance SET out_time = ?, duration = ? WHERE date = ? AND id = ?',
                       (exit_stamp, time_diff, day_key, user_id))  # Update
        conn.commit()
        conn.close()
        logging.info(f"Checkout for {user_id}: {time_diff} minutes")  # Test log
        return jsonify({'message': 'Checkout complete', 'duration': time_diff})
    logging.warning(f"No open entry for {user_id}")
    conn.close()
    return jsonify({'error': 'No open entry'})

@app.route('/retrieve_daily_report/<report_date>', methods=['GET'])  # Unique param
def retrieve_daily_report(report_date):
    """Main action: Fetch and analyze daily records.
    Sub-action: Query records.
    Testable: Log rate.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, present, in_time, out_time, duration FROM daily_attendance WHERE date = ?', (report_date,))
    daily_list = [{'id': row[0], 'present': bool(row[1]), 'in_time': row[2], 'out_time': row[3], 'duration': row[4]} for row in cursor.fetchall()]  # Collect
    conn.close()
    total_items = len(daily_list)  # Count total
    present_items = sum(1 for item in daily_list if item['present'])  # Count present
    presence_ratio = (present_items / total_items * 100) if total_items > 0 else 0  # Calc rate
    logging.info(f"Report {report_date}: {present_items}/{total_items} present ({presence_ratio:.1f}%)")  # Test log
    return jsonify({'date': report_date, 'records': daily_list, 'present_rate': f"{presence_ratio:.1f}%"})

@app.route('/add_student_profile', methods=['POST'])  # Unique endpoint
def add_student_profile():
    """Main action: Enroll new student with 100 images, retrain.
    Sub-action: Validate form data.
    Testable: Log img count, accuracy.
    """
    if 'new_photos' not in request.files or 'profile_info' not in request.form:  # Sub-action: Check files/form
        logging.error("Missing photos or info in enrollment.")
        return jsonify({'error': 'Missing data'}), 400

    info_str = request.form['profile_info']  # Get info
    profile_dict = json.loads(info_str)  # Parse JSON
    student_name = profile_dict.get('full_name', '').strip()  # Extract name
    student_code = profile_dict.get('id', '').strip()  # Extract code
    student_gender = profile_dict.get('gender', 'Unknown')  # Extract gender

    if not student_name or not student_code:  # Sub-action: Validate details
        logging.error("Incomplete student details.")
        return jsonify({'error': 'Incomplete details'}), 400

    new_subfolder = f"{student_name.replace(' ', '_')}_{student_code}"  # Unique folder name
    new_full_path = os.path.join(data_store, new_subfolder)  # Path
    os.makedirs(new_full_path, exist_ok=True)  # Create dir
    photo_counter = 0  # Counter

    for photo in request.files.getlist('new_photos'):  # Sub-action: Loop photos
        if photo.filename.lower().endswith(('.jpg', '.jpeg')):  # Filter
            buffer_data = np.frombuffer(photo.read(), np.uint8)  # Buffer
            decoded_photo = cv2.imdecode(buffer_data, cv2.IMREAD_COLOR)  # Decode
            valid_photo_crop = find_valid_face(decoded_photo)  # Detect/validate
            if valid_photo_crop is not None:
                saved_photo_name = os.path.join(new_full_path, f"enrolled_gray_{new_subfolder}_{photo_counter+1}.jpg")  # Unique name
                cv2.imwrite(saved_photo_name, valid_photo_crop)  # Save
                photo_counter += 1
                logging.info(f"Enrolled photo {photo_counter} for {student_name}")  # Log

    if photo_counter < 100:  # Sub-action: Count check
        logging.error(f"Insufficient photos {photo_counter} for {student_name}.")
        return jsonify({'error': f'Only {photo_counter} photos; require 100'}), 400

    # Sub-action: Save to SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO student_profiles (id, full_name, gender, face_folder) VALUES (?, ?, ?, ?)',
                   (student_code, student_name, student_gender, new_subfolder))  # Set doc
    conn.commit()
    conn.close()
    logging.info(f"Profile saved for {student_name} with {photo_counter} photos")

    # Helper: Gather features and labels from dataset
    def gather_dataset_features(dataset_path, image_size):
        """Load images, extract HOG features, and return feature and label lists."""
        features = []
        labels = []
        for student_folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, student_folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            resized_img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
                            hog_feat = get_hog_desc(resized_img)
                            features.append(hog_feat)
                            labels.append(student_folder.split('_')[0])  # Use name part as label
        return features, labels

    # Sub-action: Retrain model
    def refresh_model(dataset_path):  # Unique inner func
        feat_list, tag_list = gather_dataset_features(dataset_path, (100, 100))  # Reload data
        if len(feat_list) == 0:  # Check empty
            logging.error("No data for refresh.")
            raise ValueError("No data for refresh.")
        feat_normalizer.fit(feat_list)  # Refit scaler
        scaled_list = feat_normalizer.transform(feat_list)  # Scale
        train_split, test_split, train_tag, test_tag = train_test_split(scaled_list, tag_list, test_size=0.2, random_state=202)  # Split
        face_recognizer.fit(train_split, train_tag)  # Refit SVM
        pred_tag = face_recognizer.predict(test_split)  # Predict
        refresh_acc = accuracy_score(test_tag, pred_tag)  # Score
        logging.info(f"Refreshed accuracy: {refresh_acc * 100:.2f}%")  # Test log
        assert refresh_acc >= 0.95, "Refreshed accuracy below threshold"  # Test
        print(f"Test: Retrain accuracy = {refresh_acc * 100:.2f}%")  # Test print
        label_key.update({t: i for i, t in enumerate(np.unique(tag_list), len(label_key))})  # Update map
        refresh_pack = {'svm_model': face_recognizer, 'feature_scaler': feat_normalizer, 'tag_dict': label_key}  # Pack
        with open(model_path, 'wb') as refresh_handle:  # Save
            pickle.dump(refresh_pack, refresh_handle)
        logging.info("Model refreshed and saved.")
        return refresh_pack

    refresh_model(data_store)  # Call retrain
    return jsonify({'status': 'enrolled', 'message': f'{student_name} added with {photo_counter} photos'})


# Health check endpoint
@app.route('/', methods=['GET'])
def ping():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    logging.info("Server starting on port 5000.")  # Startup log
    app.run(debug=True, port=5000, host='0.0.0.0')  # Run with host for accessibility


