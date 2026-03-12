
import modal
import subprocess

app = modal.App("osf-list")
image = modal.Image.debian_slim().pip_install(["osfclient"])

@app.function(image=image)
def list_files(project_id):
    print(f"Listing files for OSF project: {project_id}")
    res = subprocess.run(["osf", "-p", project_id, "list"], capture_output=True, text=True)
    print(res.stdout)

@app.local_entrypoint()
def main():
    list_files.remote("q3zws")
    list_files.remote("2urht")
