import csv
import random
import string

def generate_random_data(num_records):
    categories = ["food", "rent", "travel", "shopping", "utilities", "entertainment"]
    random.shuffle(categories)

    data = []
    for _ in range(num_records):
        record = {
            "dat": generate_random_date(),
            "id": generate_random_transaction_id(),
            "amo": round(random.uniform(-1000, 1000), 2),
            "cat": categories[_ % len(categories)],
            "des": generate_random_description()
        }
        data.append(record)
    return data

def generate_random_date():
    year = random.randint(2010, 2023)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"

def generate_random_transaction_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def generate_random_description():
    descriptions = ["payment", "purchase", "expense", "refund", "deposit"]
    return random.choice(descriptions)

# Generate 100 random financial records
num_records = 100
financial_data = generate_random_data(num_records)

# Save data to a CSV file
filename = "financial_data.csv"
with open(filename, "w", newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["dat", "id", "amo", "cat", "des"])
    writer.writeheader()
    writer.writerows(financial_data)

print(f"Generated {num_records} records and saved to {filename}.")
