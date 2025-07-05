import sqlite3
import pandas as pd
from typing import List, Optional, Tuple
import datetime

# Define database file names
db_file_regular = "vehicles.db"
db_file_detected_log = "detected_vehicles_log.db"

def create_regular_vehicles_table():
    """Creates the 'vehicles' table in the regular database with multiple photo columns."""
    try:
        with sqlite3.connect(db_file_regular) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_name TEXT,
                    number_plate TEXT UNIQUE,
                    vehicle_type TEXT,
                    vehicle_model TEXT,
                    colour TEXT,
                    owner_face_photo1 BLOB,
                    owner_face_photo2 BLOB,
                    owner_face_photo3 BLOB,
                    vehicle_photo1 BLOB,
                    vehicle_photo2 BLOB,
                    vehicle_photo3 BLOB,
                    vehicle_photo4 BLOB,
                    vehicle_photo5 BLOB,
                    vehicle_photo6 BLOB
                )
            """)
        print(f"Table 'vehicles' created successfully in '{db_file_regular}'.")
    except sqlite3.Error as e:
        print(f"Error creating table in {db_file_regular}: {e}")

def create_detected_vehicles_log_table():
    """
    Creates the 'detected_vehicles_log' table, which now logs all detected vehicles
    and includes a column to indicate if they were matched in the regular database.
    """
    try:
        with sqlite3.connect(db_file_detected_log) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DROP TABLE IF EXISTS detected_vehicles_log;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_vehicles_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_number_plate TEXT UNIQUE,      -- Number plate as detected
                    database_number_plate TEXT,             -- Matched number plate from DB (if any)
                    vehicle_id_db INTEGER,                  -- ID from regular_vehicles_db (if matched)
                    owner_name_db TEXT,                     -- Owner name from DB (if matched)
                    vehicle_type_db TEXT,                   -- Vehicle type from DB (if matched)
                    color_db TEXT,                          -- Vehicle color from DB (if matched)
                    detection_timestamp TEXT,               --YYYY-MM-DD HH:MM:SS
                    is_matched_in_regular_db TEXT,          -- 'Yes' or 'No'
                    detection_category TEXT,                -- e.g., 'Registered', 'Unregistered/Unknown', 'Behavioral Anomaly'
                    details TEXT,                           -- Specific notes on why it was logged
                    detection_method TEXT,                  -- e.g., 'System Lookup', 'AI-Camera', 'Manual Report'
                    current_status TEXT,                    -- e.g., 'Normal', 'Monitoring', 'Alerted Police', 'Cleared'
                    vehicle_image_detected BLOB,            -- Vehicle image captured at detection
                    vehicle_image_database BLOB,            -- Vehicle image from regular DB (if matched, e.g., photo1)
                    face_image_detected BLOB,               -- Face image captured at detection
                    face_image_database BLOB                -- Face image from regular DB (if matched, e.g., photo1)
                )
            """)
        print(f"Table 'detected_vehicles_log' created successfully in '{db_file_detected_log}'.")
    except sqlite3.Error as e:
        print(f"Error creating table in {db_file_detected_log}: {e}")

def read_image_data(file_path: Optional[str]) -> Optional[bytes]:
    """Reads image data from the given file path in binary mode."""
    if not file_path:
        return None
    try:
        with open(file_path, 'rb') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def add_new_regular_vehicle(owner: str, plate: str, v_type: str, model: str, color: str,
                            owner_face_paths: Optional[List[str]] = None,
                            vehicle_photo_paths: Optional[List[str]] = None):
    """Adds a new vehicle's information to the regular 'vehicles' table with multiple photos."""

    owner_face_data = [read_image_data(path) for path in (owner_face_paths or [])[:3]]
    owner_face_data.extend([None] * (3 - len(owner_face_data)))

    vehicle_photo_data = [read_image_data(path) for path in (vehicle_photo_paths or [])[:6]]
    vehicle_photo_data.extend([None] * (6 - len(vehicle_photo_data)))

    try:
        with sqlite3.connect(db_file_regular) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vehicles (owner_name, number_plate, vehicle_type, vehicle_model, colour,
                                     owner_face_photo1, owner_face_photo2, owner_face_photo3,
                                     vehicle_photo1, vehicle_photo2, vehicle_photo3,
                                     vehicle_photo4, vehicle_photo5, vehicle_photo6)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (owner, plate, v_type, model, color, *owner_face_data, *vehicle_photo_data))
        print(f"Vehicle '{plate}' added successfully to regular database.")
    except sqlite3.IntegrityError:
        print(f"Error: Vehicle with number plate '{plate}' already exists in regular database.")
    except sqlite3.Error as e:
        print(f"Error adding vehicle to regular database: {e}")

def add_detected_vehicle_log(
    detected_plate: str,
    database_plate: Optional[str],
    vehicle_id_db: Optional[int],
    owner_name_db: Optional[str],
    type_db: Optional[str],
    color_db: Optional[str],
    is_matched_in_regular_db: str, # 'Yes' or 'No'
    detection_category: str,
    details: str,
    detection_method: str,
    current_status: str,
    vehicle_image_detected_data: Optional[bytes] = None,
    vehicle_image_database_data: Optional[bytes] = None,
    face_image_detected_data: Optional[bytes] = None,
    face_image_database_data: Optional[bytes] = None
):
    """Adds a new detected vehicle's information to the 'detected_vehicles_log' table."""
    detection_timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds') #YYYY-MM-DD HH:MM:SS

    try:
        with sqlite3.connect(db_file_detected_log) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detected_vehicles_log (
                    detected_number_plate, database_number_plate, vehicle_id_db, owner_name_db,
                    type_db, color_db, detection_timestamp, is_matched_in_regular_db,
                    detection_category, details, detection_method, current_status,
                    vehicle_image_detected, vehicle_image_database,
                    face_image_detected, face_image_database
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (detected_plate, database_plate, vehicle_id_db, owner_name_db,
                  type_db, color_db, detection_timestamp, is_matched_in_regular_db,
                  detection_category, details, detection_method, current_status,
                  vehicle_image_detected_data, vehicle_image_database_data,
                  face_image_detected_data, face_image_database_data))
        print(f"Vehicle '{detected_plate}' logged successfully in detected vehicles log.")
    except sqlite3.IntegrityError:
        print(f"Error: Vehicle '{detected_plate}' already exists in detected vehicles log (unique number plate constraint).")
    except sqlite3.Error as e:
        print(f"Error adding vehicle to detected vehicles log: {e}")

def lookup_vehicle_in_regular_db(number_plate: str) -> Optional[Tuple]:
    """
    Checks if a vehicle with the given number plate exists in the regular database.
    Returns the row data if found, otherwise None.
    """
    try:
        with sqlite3.connect(db_file_regular) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, owner_name, number_plate, vehicle_type, colour, vehicle_photo1, owner_face_photo1
                FROM vehicles WHERE number_plate = ?
            """, (number_plate,))
            result = cursor.fetchone()
            return result # Returns (id, owner_name, number_plate, vehicle_type, colour, vehicle_photo1, owner_face_photo1) or None
    except sqlite3.Error as e:
        print(f"Error looking up vehicle in regular database: {e}")
        return None

def simulate_vehicle_detection():
    """
    Simulates vehicle detection. All detected vehicles are logged in detected_vehicles_log.db,
    with a flag indicating if they were matched in the regular database.
    """
    print("\n--- Simulate Vehicle Detection ---")
    detected_number_plate = input("Enter vehicle number plate detected by system: ").strip('"') 

    db_match_data = lookup_vehicle_in_regular_db(detected_number_plate)
    
    is_matched_status = 'Yes' if db_match_data else 'No'

    # Initialize data from database as None
    database_number_plate = None
    vehicle_id_db = None
    owner_name_db = None
    type_db = None
    color_db = None
    vehicle_image_database_data = None
    face_image_database_data = None

    # Mock/Placeholder data for detected images/details (comes from actual detection system)
    # In a real system, these would be actual captured photos/readings
    vehicle_image_detected_data = None # e.g., read_image_data("path/to/detected_vehicle.jpg")
    face_image_detected_data = None    # e.g., read_image_data("path/to/detected_face.jpg")


    if db_match_data:
        # Extract matched data from database row
        (vehicle_id_db, owner_name_db, database_number_plate, type_db, color_db,
         vehicle_image_database_data, face_image_database_data) = db_match_data

        print(f"Vehicle '{detected_number_plate}' detected. Matched in regular database.")
        detection_category = "Registered"
        details = "Vehicle found in regular database."
        current_status = "Normal Operation"
    else:
        print(f"Vehicle '{detected_number_plate}' detected. NOT matched in regular database. Flagged as Unregistered/Unknown.")
        detection_category = "Unregistered/Unknown"
        details = "Vehicle number plate not found in registered database."
        current_status = "Monitoring"

        # if ai_model.detect_anomaly(detected_image_data):
        #     detection_category = "Behavioral Anomaly"
        #     details = "Detected unusual movement pattern by AI camera."
        #     current_status = "Alerted Police"

    add_detected_vehicle_log(
        detected_plate=detected_number_plate,
        database_plate=database_number_plate,
        vehicle_id_db=vehicle_id_db,
        owner_name_db=owner_name_db,
        type_db=type_db,
        color_db=color_db,
        is_matched_in_regular_db=is_matched_status,
        detection_category=detection_category,
        details=details,
        detection_method="Simulated Camera/System Lookup", # Combined method for clarity
        current_status=current_status,
        vehicle_image_detected_data=vehicle_image_detected_data,
        vehicle_image_database_data=vehicle_image_database_data,
        face_image_detected_data=face_image_detected_data,
        face_image_database_data=face_image_database_data
    )


def view_regular_vehicles():
    """Fetches and displays all regular vehicles from the database using Pandas."""
    try:
        with sqlite3.connect(db_file_regular) as conn:
            query = "SELECT id, owner_name, number_plate, vehicle_type, vehicle_model, colour FROM vehicles"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                print("\n--- Regular Vehicles ---")
                print(df.to_string(index=False))
            else:
                print("\nNo regular vehicles found in the database.")
    except Exception as e:
        print(f"Error viewing regular vehicles: {e}")


def view_detected_vehicle_log():
    """Fetches and displays all detected vehicles from the log database using Pandas."""
    try:
        with sqlite3.connect(db_file_detected_log) as conn:
            query = """
            SELECT id, detected_number_plate, database_number_plate, vehicle_id_db,
                   owner_name_db, type_db, color_db, detection_timestamp, is_matched_in_regular_db,
                   detection_category, details, detection_method, current_status
            FROM detected_vehicles_log
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                print("\n--- Detected Vehicles Log ---")
                print(df.to_string(index=False))
            else:
                print("\nNo detected vehicles found in the log database.")
    except Exception as e:
        print(f"Error viewing detected vehicles log: {e}")


def analyze_regular_data():
    """Retrieves regular vehicle data and analyzes it using pandas."""
    try:
        with sqlite3.connect(db_file_regular) as conn:
            query = "SELECT vehicle_type, colour FROM vehicles"
            df = pd.read_sql_query(query, conn)

        if df.empty:
            print("\nNo regular vehicle data to analyze.")
            return

        print("\n--- Regular Vehicle Type Counts (Pandas) ---")
        print(df['vehicle_type'].value_counts().to_string()) # Added .to_string()
        print("\n--- Regular Colour Counts (Pandas) ---")
        print(df['colour'].value_counts().to_string()) # Added .to_string()

    except Exception as e:
        print(f"Error analyzing regular data: {e}")

def analyze_detected_vehicle_log_data():
    """Retrieves detected vehicle log data and analyzes it using pandas."""
    try:
        with sqlite3.connect(db_file_detected_log) as conn:
            query = """
            SELECT is_matched_in_regular_db, detection_category, detection_method, current_status
            FROM detected_vehicles_log
            """
            df = pd.read_sql_query(query, conn)

        if df.empty:
            print("\nNo detected vehicle log data to analyze.")
            return

        print("\n--- Match Status Counts (Pandas) ---")
        print(df['is_matched_in_regular_db'].value_counts().to_string()) # Added .to_string()
        print("\n--- Detection Category Counts (Pandas) ---")
        print(df['detection_category'].value_counts().to_string()) 
        print("\n--- Detection Method Counts (Pandas) ---")
        print(df['detection_method'].value_counts().to_string()) 
        print("\n--- Current Status Counts (Pandas) ---")
        print(df['current_status'].value_counts().to_string()) 

    except Exception as e:
        print(f"Error analyzing detected vehicle log data: {e}")


def main():
    """Main function to manage vehicle and detected vehicle log databases."""
    # Ensure both tables are created when the program starts
    create_regular_vehicles_table()
    create_detected_vehicles_log_table() # This will now ensure the table has the correct schema

    while True:
        print("\n--- Main Menu ---")
        print("1. Add a Regular Vehicle")
        print("2. Simulate Vehicle Detection (Records All Detections)")
        print("3. View All Regular Vehicles")
        print("4. View All Detected Vehicle Log Entries")
        print("5. Analyze Regular Vehicle Data")
        print("6. Analyze Detected Vehicle Log Data")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("\n--- Add Regular Vehicle ---")
            owner_name = input("Enter owner's name: ")
            number_plate = input("Enter vehicle number plate: ").strip('"')
            vehicle_type = input("Enter vehicle type: ")
            vehicle_model = input("Enter vehicle model: ")
            colour = input("Enter vehicle colour: ")

            add_photo_choice = input("Do you want to add owner and vehicle photos now? (yes/no): ").lower()
            owner_face_paths = []
            vehicle_photo_paths = []

            if add_photo_choice == 'yes':
                print("Enter paths for owner's face photos (up to 3):")
                for i in range(1, 4):
                    path = (input(f"  Face photo {i} path (leave blank if none): ") or "").strip('"') or None
                    if path:
                        owner_face_paths.append(path)

                print("\nEnter paths for vehicle photos (up to 6):")
                for i in range(1, 7):
                    path = (input(f"  Vehicle photo {i} path (leave blank if none): ") or "").strip('"') or None
                    if path:
                        vehicle_photo_paths.append(path)
            
            add_new_regular_vehicle(owner_name, number_plate, vehicle_type, vehicle_model, colour, owner_face_paths, vehicle_photo_paths)

        elif choice == '2':
            simulate_vehicle_detection()

        elif choice == '3':
            view_regular_vehicles()

        elif choice == '4':
            view_detected_vehicle_log()

        elif choice == '5':
            analyze_regular_data()

        elif choice == '6':
            analyze_detected_vehicle_log_data()

        elif choice == '7':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()