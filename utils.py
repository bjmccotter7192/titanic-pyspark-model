import json
import pandas as pd
import string
import random
from pyspark.sql.functions import col, when

FILE_NAME = "random_data.csv"

def generate_random_name():
    # Generate unique first names
    first_names = [
        "Emma", "Liam", "Olivia", "Noah", "Ava", "Isabella", "Sophia", "Jackson", "Mia", "Lucas",
        "Oliver", "Aiden", "Sophia", "Amelia", "Henry", "Charlotte", "Ethan", "Luna", "Mia", "Liam",
        "Ella", "Oliver", "Ava", "Harper", "Evelyn", "Benjamin", "Emily", "Isabella", "Sebastian", "Michael",
        "Elena", "Chloe", "Grace", "Carter", "Aria", "Scarlett", "Aiden", "Liam", "Madison", "Abigail",
        "Charlotte", "Harper", "Ella", "Evelyn", "Lily", "Amelia", "Olivia", "Ava", "Mia", "Sophia",
        "Jackson", "Emma", "Liam", "Oliver", "Noah", "Aiden", "Sophia", "Isabella", "Mia", "Lucas",
        "Olivia", "Ella", "Isabella", "Ava", "Mia", "Sophia", "Amelia", "Jackson", "Evelyn", "Harper",
        "Liam", "Oliver", "Noah", "Ethan", "Emma", "Sophia", "Ava", "Olivia", "Isabella", "Mia",
        "Liam", "Noah", "Oliver", "Elijah", "Emma", "Olivia", "Ava", "Sophia", "Isabella", "Mia",
        "Liam", "Noah", "Oliver", "Elijah", "Emma", "Sophia", "Isabella", "Ava", "Mia", "Olivia"
    ]

    # Shuffle the first names to ensure randomness
    random.shuffle(first_names)

    # Generate unique middle initials (single uppercase letters)
    middle_initials = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Shuffle the middle initials to ensure randomness
    random.shuffle(middle_initials)

    # Generate unique last names
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
        "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King",
        "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter",
        "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins",
        "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey",
        "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres", "Peterson", "Gray", "Ramirez",
        "James", "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross",
        "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson", "Hughes", "Flores", "Washington"
    ]

    # Shuffle the last names to ensure randomness
    random.shuffle(last_names)

    return f"{random.choice(first_names)} {random.choice(middle_initials)}. {random.choice(last_names)}"

def generate_sample_dataset(num_rows=10000):
    print("Starting to create your random data set")
    data = {
        'PassengerId': list(range(1, num_rows + 1)),
        'Survived': [int(random.choice([0, 1])) for _ in range(num_rows)],
        'Pclass': [int(random.choice([1, 2, 3])) for _ in range(num_rows)],
        'Name': [generate_random_name() for _ in range(num_rows)],
        'Sex': [random.choice(['male', 'female']) for _ in range(num_rows)],
        'Age': [float(round(random.uniform(1, 89), 2)) for _ in range(num_rows)],
        'SibSp': [int(random.randint(1, 5)) for _ in range(num_rows)],
        'Parch': [int(random.randint(1, 5)) for _ in range(num_rows)],
        'Ticket': [''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(num_rows)],
        'Fare': [float(round(random.uniform(10, 1000), 2)) for _ in range(num_rows)],
        'Cabin': [random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G']) for _ in range(num_rows)],
        'Embarked': [random.choice(['S', 'C', 'Q']) for _ in range(num_rows)]
    }

    df = pd.DataFrame(data)
    df.to_csv(FILE_NAME, index=False)
    print(f"Sample dataset saved to {FILE_NAME}")

def generate_extra_cols(df):
    # Title from Name
    df = df.withColumn('Title', when(col('Name').like('%Mr.%'), 'Mr')
                                .when(col('Name').like('%Mrs.%'), 'Mrs')
                                .when(col('Name').like('%Miss.%'), 'Miss')
                                .otherwise('Other'))

    # Family Size
    df = df.withColumn('FamilySize', col('SibSp') + col('Parch') + 1)

    # Age Group
    df = df.withColumn('AgeGroup', when(col('Age') < 18, 'Child')
                                    .when((col('Age') >= 18) & (col('Age') < 65), 'Adult')
                                    .otherwise('Elderly'))

    # Cabin Deck
    df = df.withColumn('Deck', when(col('Cabin').isNotNull(), col('Cabin').substr(0, 1)).otherwise('Unknown'))

    # # Ticket Prefix
    # df = df.withColumn('TicketPrefix', when(col('Ticket').rlike('[A-Za-z]'), col('Ticket').regexp_extract(r'([A-Za-z]+)', 1)).otherwise('Unknown'))

    # Is Alone
    df = df.withColumn('IsAlone', when(col('FamilySize') == 1, 1).otherwise(0))

    return df