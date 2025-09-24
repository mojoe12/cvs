import os
import subprocess
import sys
import venv

def create_and_setup_venv():
    venv_dir = "venv"
    print(f"Creating virtual environment in {venv_dir}...")
    
    # Create virtual environment
    venv.create(venv_dir, with_pip=True)
    
    # Determine the pip executable path
    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
    
    # Install dependencies from requirements.txt
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call([pip_path, "install", "-r", requirements_file])
    else:
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)
    
    print("Virtual environment setup complete!")
    print("To activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir}/bin/activate")

if __name__ == "__main__":
    create_and_setup_venv()
