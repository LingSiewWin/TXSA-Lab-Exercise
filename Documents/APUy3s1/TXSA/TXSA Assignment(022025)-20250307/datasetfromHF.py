from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    repo_id="FerasZawahreh/Employees_Reviews_For_Their_Company_DS",
    filename="Employees_Reviews_For_Their_Company_DS.csv",
    repo_type="dataset"
)

print("File saved at:", csv_path)
