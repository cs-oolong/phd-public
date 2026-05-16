import sys
import subprocess
import requests
import os

arxiv_id = sys.argv[1]
pdf_path = f"/tmp/{arxiv_id}.pdf"
txt_path = f"/tmp/{arxiv_id}.txt"
archive_dir = os.path.expanduser("~/monorepo/phd/papers")

print(f"Fetching arXiv:{arxiv_id}...")

# Download PDF
url = f"https://arxiv.org/pdf/{arxiv_id}"
r = requests.get(url)
if r.status_code != 200:
    print(f"Failed to download: {r.status_code}")
    sys.exit(1)
open(pdf_path, "wb").write(r.content)
print(f"Downloaded PDF ({len(r.content)} bytes)")

# Convert to text
try:
    subprocess.run(["pdftotext", pdf_path, txt_path], check=True)
    print(f"Converted to text")
except FileNotFoundError:
    print("ERROR: pdftotext not found. Install with: sudo apt-get install poppler-utils")
    sys.exit(1)

# Archive
os.makedirs(archive_dir, exist_ok=True)
os.replace(pdf_path, f"{archive_dir}/{arxiv_id}.pdf")
os.replace(txt_path, f"{archive_dir}/{arxiv_id}.txt")

print(f"Saved to {archive_dir}/{arxiv_id}.pdf")
print(f"Saved to {archive_dir}/{arxiv_id}.txt")
