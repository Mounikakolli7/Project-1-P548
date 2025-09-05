import os
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader

# Path to your dataset folder (update if needed)
DATASET_DIR = "Resumes_Docx"

# Empty lists
texts = []
labels = []

# Loop through each subfolder
for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(folder_path):  # only process folders
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            text = ""
            if file.endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)
            elif file.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            if text.strip():  # only if text is not empty
                texts.append(text)
                labels.append(folder)  # folder name = category

# Save to CSV
df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("dataset.csv", index=False, encoding="utf-8")

print("✅ dataset.csv created successfully with", len(df), "records")
