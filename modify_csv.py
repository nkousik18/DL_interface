import pandas as pd

# Path to your CSV
csv_path = "id_to_name.csv"
backup_path = "id_to_name_backup.csv"

# Load the CSV
df = pd.read_csv(csv_path)
print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
print("[DEBUG] Columns:", list(df.columns))

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Check required columns
if "Student Name" not in df.columns or "Celebrity Choice/ID" not in df.columns:
    raise ValueError("❌ CSV must contain 'Student Name' and 'Celebrity Choice/ID' columns.")

# Create new email column
emails = []
for _, row in df.iterrows():
    name = str(row["Student Name"]).strip()
    celeb_id = str(row["Celebrity Choice/ID"]).strip()
    first_name = name.split()[0].lower() if name else "user"
    email = f"{first_name}{celeb_id}@gmail.com"
    emails.append(email)

df["Email"] = emails

# Backup old file
df.to_csv(backup_path, index=False)
print(f"[INFO] Backup saved as {backup_path}")

# Overwrite or save new CSV
updated_path = "id_to_name_updated.csv"
df.to_csv(updated_path, index=False)
print(f"[INFO] ✅ Updated CSV with 'Email' column saved as {updated_path}")

print(df.head())
