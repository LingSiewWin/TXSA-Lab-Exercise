from huggingface_hub import hf_hub_download
import pandas as pd

# Download the CSV file to a local path
csv_path = hf_hub_download(
    repo_id="FerasZawahreh/Employees_Reviews_For_Their_Company_DS",
    filename="Employees_Reviews_For_Their_Company_DS.csv",
    repo_type="dataset"
)

# Now read the CSV using pandas
df = pd.read_csv(csv_path)

print(df.head())
