# Getting Started with This Project

Hey there! So you've cloned the repository – awesome! Let's get you set up to run this project. We use a virtual environment to keep things tidy and ensure everyone is working with the same library versions. This repository contains **three demo projects**, each in its own folder at the root of the repository. Each demo has its own `requirements.txt` for dependencies.

## Setting Up Your Local Environment

**1. Creating the Virtual Environment (if it doesn't exist):**

If you don't see a `venv` folder in the project root, create one by running:

```
python -m venv venv
```

**2. Activating the Virtual Environment:**

Activate the virtual environment before installing dependencies or running any demo.

* **Windows (Command Prompt):**
    ```
    venv\Scripts\activate
    ```
* **Windows (PowerShell):**
    ```
    .\venv\Scripts\Activate.ps1
    ```
* **Linux/macOS:**
    ```
    source venv/bin/activate
    ```

Once activated, you'll see `(venv)` at the start of your terminal prompt.

---

## Working with the Demo Projects

There are three folders in the root directory, each containing a separate demo project. For each demo, follow these steps:

### 1. Change Directory to the Demo

Replace `<demo-folder>` with the actual folder name (e.g., `local-hello-worldrag`, `llama3-base`, `gemini-rag-hello-world`):

```
cd <demo-folder>
```

### 2. Install Dependencies

Each demo has its own `requirements.txt`. With your virtual environment activated and inside the demo folder, run:

```
pip install -r requirements.txt
```

### 3. Run the Demo

Check the `README.md` or documentation inside each demo folder for specific instructions on how to start that demo. Common ways to start a Python project include:

```
python main.py
```
or
```
python app.py
```

---

## Example Workflow

Here's how you might run a demo called `demo1`:

```
# From the project root
python -m venv venv
venv\Scripts\activate         # or source venv/bin/activate on Linux/macOS
cd demo1
pip install -r requirements.txt
python main.py                # or the appropriate start command for the demo
```

---

## Summary

- Use a virtual environment (`venv`) in the project root.
- Each demo project is in its own folder with its own `requirements.txt`.
- Activate the virtual environment, change into the demo folder, install dependencies, and run the demo as instructed.

Happy coding!