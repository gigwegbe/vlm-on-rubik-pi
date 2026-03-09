# Visual Language Model on Rubik-Pi

## 1. Create a Python Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv vlmenv
source vlmenv/bin/activate
```

## 2. Install Dependencies

Install the required Python packages:

  ```
  pip install -r requirements.txt 
  ```

## 3. Download the Model

Download the required model from Hugging Face (replace with the correct repository):

  ```
  hfdownload <model-repository>
  ```

## 4. Start the Inference Engine

Launch the llama.cpp inference server:

  ```
    ./start_llama.cpp
  ```

Make sure the script has execution permission:

  ```
    chmod +x start_llama.cpp
  ```

## 5. Start the Application

  ```
  python3 app.py   
  ```
