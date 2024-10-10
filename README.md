# MLOps-Project

## Project 1: Face Recognition

In this project, we focus on MLOps by developing a face recognition system. The objective is to allow users to enroll in the system by taking a photograph of their face. During subsequent logins, their faces will be verified before granting access. We will utilize a pre-trained model from the MTCNN library for this purpose.

## User Instructions
- Create a new folder face-recognition-system
- Clone the GitHub repository
git clone < paste the url here >
cd face-recognition-system
### Create a virtual environment
  * Install the virtual environment: pip install virtualenv
  * run: python -m < your-virtual-env-name >
  * Activate your virtual environment: source < your-virtual-env-name >/bin/activate # (On windows use) < your-virtual-env-name >\Scripts\activate
  * To deactivate the virtual environment, run: deactivate
- Run the requirements.txt file as it contains the Python dependencies that we will need for the project.
  * run: pip install -r requirements.txt
  * Run the main.py file: python main.py

 ### Deployment
  * API Development with FAST API
  * In the main.py file, we have API endpoints for user enrollment, login, updates, deletions, and face recognition. Run the API using Uvicorn:
  * uvicorn main:app --reload
  * Copy this (http://127.0.0.1:8000/docs), open your browser and paste it.
### Docker
- Ensure you have Docker installed, you can install it from [here](https://docs.docker.com/desktop/?_gl=1*wtu5yy*_gcl_au*MTcwMDA1NDUzMi4xNzI4MTI3ODE0*_ga*MzI4MDQwOTk1LjE3MjcyODA5OTg.*_ga_XJWPQMJYHQ*MTcyODEyNzc4Ny4zLjEuMTcyODEyNzgxNC4zMy4wLjA.).
  * Build the docker image and run the docker container.
    
    docker build -t face-recognition-system .  (copy and paste up to the dot)
    
    docker run -d -p 8000:8000 face-recognition-system
  * Open (http://localhost:8000) on your browser to access the FastAPI application.
 ### Data Versioning
 - Data versioning with DVC
   * Install [DVC](https://dvc.org/doc/install), if not yet installed.
  
     dvc init
   * Track data files assuming your dataset is in a folder named data.
     
     dvc add data/
   * Push data to a remote storage
     
     dvc remote add -d myremote <remote_storage_url>  #the remote storage could be AWS, GCP, Azure etc
     
     dvc push
 ### Frontend with Plotly Dash
- It provides an easy way to create a frontend interface for users to interact with the brain tumour segmentation service. We chose Plotly Dash as it is a powerful framework that's well-suited for data visualizations given that we will be using image data.
  * Create a plotly dash script, e.g app.py
  * Run the Plotly Dash application
    
    pip install dash
    
    python app.py
    
 * After running the command, you will see an output ( http://127.0.0.1:8050/ or http://localhost:8050/ ).
 * Open your browser and paste the URL to access the dash application.

