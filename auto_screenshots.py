import os
import time
import subprocess
import shutil
from PIL import Image, ImageDraw, ImageFont
import git
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_placeholder_image(filepath, text="Screenshot needed"):
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    # Calculate text bounding box
    bbox = d.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((800 - text_width) / 2, (600 - text_height) / 2)
    d.text(position, text, fill=(0, 0, 0), font=font)
    img.save(filepath)

def main():
    print("--- Starting Auto-Screenshot Process ---")
    
    images_dir = "images"
    ensure_dir(images_dir)

    # 1. Start services
    print("\nStarting services...")
    flask_process = None
    streamlit_process = None

    if os.path.exists("src/flask_api.py"):
        try:
            flask_process = subprocess.Popen(["python", "src/flask_api.py"])
            print("Started Flask API.")
        except Exception as e:
            print(f"Failed to start Flask API: {e}")
    else:
        print("Flask API (src/flask_api.py) not found. Skipping.")

    try:
        streamlit_process = subprocess.Popen(["streamlit", "run", "dashboard.py", "--server.headless=true"])
        print("Started Streamlit dashboard.")
    except Exception as e:
        print(f"Failed to start Streamlit: {e}")

    print("Waiting 10 seconds for services to start...")
    time.sleep(10) # Increased to 10 seconds to ensure streamlit is fully loaded

    # 2. Take screenshots using Selenium
    print("\nTaking Streamlit dashboard screenshot...")
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get("http://localhost:8501")
        print("Waiting 5 seconds for page to fully render...")
        time.sleep(5)
        
        dashboard_path = os.path.join(images_dir, "dashboard_final.png")
        driver.save_screenshot(dashboard_path)
        print(f"Saved dashboard screenshot to {dashboard_path}")
        
        driver.quit()
    except Exception as e:
        print(f"Selenium failed to capture dashboard: {e}")
        print("Please ensure you have Chrome installed and internet access for webdriver_manager.")

    # Terminate services
    print("\nTerminating background services...")
    if flask_process:
        flask_process.terminate()
    if streamlit_process:
        streamlit_process.terminate()

    # 3. Handle static images
    print("\nHandling static images...")
    static_images = [
        "failure_graph.png", 
        "prediction_output.png", 
        "preprocessing.png", 
        "repo_preview.png", 
        "training_logs.png"
    ]
    
    cm_source = os.path.join("outputs", "figures", "confusion_matrix.png")
    cm_dest = os.path.join(images_dir, "confusion_matrix.png")
    
    if os.path.exists(cm_source):
        shutil.copy2(cm_source, cm_dest)
        print(f"Copied confusion matrix to {cm_dest}")
    else:
        print("Confusion matrix not found in outputs/figures/. Checking images/ folder.")
        if not os.path.exists(cm_dest):
            create_placeholder_image(cm_dest, "Screenshot needed: Confusion Matrix")
            print(f"WARNING: Created placeholder for {cm_dest}. Manual capture required.")

    for img_name in static_images:
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            create_placeholder_image(img_path, f"Screenshot needed: {img_name}")
            print(f"WARNING: Created placeholder for {img_path}. Manual capture required.")
        else:
            print(f"Found existing image: {img_path}")

    # 4. Commit and push
    print("\nCommitting and pushing to GitHub...")
    try:
        repo = git.Repo(".")
        repo.git.add(images_dir)
        repo.index.commit("Auto-update screenshots for GitHub")
        origin = repo.remote(name='origin')
        origin.push()
        print("Successfully pushed screenshots to origin.")
    except Exception as e:
        print(f"Git operation failed: {e}")
        print("Please ensure git is initialized, GitPython is installed, and remote is configured.")

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()
