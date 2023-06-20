import json
import random
import string

def generate_random_data(num_records):
    data = []
    for _ in range(num_records):
        record = {
            "date": generate_random_date(),
            "transaction_id": generate_random_transaction_id(),
            "amount": round(random.uniform(-1000, 1000), 2),
            "category": generate_random_category(),
            "description": generate_random_description()
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

def generate_random_category():
    categories = ["food", "rent", "travel", "shopping", "utilities", "entertainment"]
    return random.choice(categories)

def generate_random_description():
    descriptions = ["Payment", "Purchase", "Expense", "Refund", "Deposit"]
    return random.choice(descriptions)

# Generate 100 random financial records
num_records = 100
financial_data = generate_random_data(num_records)

# Save data to a JSON file
filename = "financial_data.json"
with open(filename, "w") as file:
    json.dump(financial_data, file, indent=4)

print(f"Generated {num_records} records and saved to {filename}.")
