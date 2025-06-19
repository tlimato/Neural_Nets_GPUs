# Author      : Tyson Limato
# Date        : 2025-6-1
# File Name   : example_data_generator.py
import csv
import random

def generate_house_data(num_samples=10000, filename="house_prices.csv"):
    """
    Generate example data for classroom teaching.
    (P.S: thanks Zillow for making real housing data impossible to get)
    """
    headers = [
        'sqft', 'beds', 'baths', 'age', 'garage', 'stories', 'lot_size',
        'distance_to_city', 'crime_rate', 'school_rating', 'has_pool',
        'has_fireplace', 'neighborhood_code', 'year_renovated', 'price'
    ]
    
    data = []
    for _ in range(num_samples):
        sqft = random.randint(800, 4000)
        beds = random.randint(1, 6)
        baths = random.randint(1, 4)
        age = random.randint(0, 50)
        garage = random.randint(0, 3)
        stories = random.randint(1, 3)
        lot_size = random.uniform(0.1, 1.0) * sqft
        distance_to_city = random.uniform(0.5, 30.0)
        crime_rate = random.uniform(0.0, 1.0)
        school_rating = random.randint(1, 10)
        has_pool = random.randint(0, 1)
        has_fireplace = random.randint(0, 1)
        neighborhood_code = random.randint(0, 5)
        year_renovated = random.choice([0] + list(range(2000, 2022)))

        # Simple formula for synthetic price
        base_price = 50000 + sqft * 150 + beds * 10000 + baths * 5000
        base_price -= age * 1000
        base_price += garage * 5000 + stories * 7000
        base_price += lot_size * 10
        base_price -= distance_to_city * 1000
        base_price += school_rating * 3000
        base_price += has_pool * 10000
        base_price += has_fireplace * 3000
        base_price += (2022 - year_renovated) * 500 if year_renovated else 0
        base_price += neighborhood_code * 8000
        price = round(base_price + random.uniform(-20000, 20000), 2)

        row = [
            sqft, beds, baths, age, garage, stories, round(lot_size, 2), round(distance_to_city, 2),
            round(crime_rate, 4), school_rating, has_pool, has_fireplace, neighborhood_code, year_renovated, price
        ]
        data.append(row)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"Dataset with {num_samples} samples saved as '{filename}'")

# Run the function
if __name__ == "__main__":
    generate_house_data()
