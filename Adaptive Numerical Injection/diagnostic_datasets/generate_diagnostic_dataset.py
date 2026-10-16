"""
Diagnostic Dataset Generator for Linear Probe Experiments

This script generates two types of datasets:
1. Contextual/Semantic: numbers used as contextual identifiers (time, date, IDs, etc.)
2. Numerical/Mathematical: numbers used in explicit mathematical operations
"""

import json
import random
import os
import re
from typing import List, Dict

# Set random seed for reproducibility
random.seed(42)


def extract_numbers_from_text(text: str) -> List[str]:
    """
    Extract all numeric substrings from the given text.
    
    Args:
        text: input text
    
    Returns:
        list of number strings
    """
    # Find number patterns (integers and decimals, including percentages)
    # Also capture numbers attached to characters (e.g., "3rd" -> "3")
    pattern = r'(?<!\d)\d+(?:\.\d+)?(?!\d)'
    numbers = re.findall(pattern, text)
    return numbers

def generate_contextual_data(num_samples: int = 1000) -> List[Dict]:
    """
    Generate contextual/semantic data where numbers act as identifiers,
    timestamps, dates, locations, etc.
    """
    data = []
    
    # Define templates
    templates = [
        # Place / room numbers
        ("The event is in Room {num}.", lambda: random.randint(100, 999)),
        ("Please go to Office {num}.", lambda: random.randint(100, 500)),
        ("The meeting is scheduled in Conference Room {num}.", lambda: random.randint(1, 50)),
        ("Her apartment is on Floor {num}.", lambda: random.randint(1, 30)),
        ("The store is located at {num} Main Street.", lambda: random.randint(1, 9999)),
        
        # Time / date
        ("He was born in {num}.", lambda: random.randint(1950, 2020)),
        ("The event took place in {num}.", lambda: random.randint(1900, 2024)),
        ("She graduated in {num}.", lambda: random.randint(1980, 2023)),
        ("The company was founded in {num}.", lambda: random.randint(1900, 2020)),
        ("The meeting is at {num} o'clock.", lambda: random.randint(1, 24)),
        ("The deadline is on the {num}th of the month.", lambda: random.randint(1, 31)),
        
        # IDs / numbers
        ("The student ID is {num}.", lambda: random.randint(10000, 99999)),
        ("The order number is {num}.", lambda: random.randint(1000, 9999)),
        ("The invoice number is {num}.", lambda: random.randint(10000, 99999)),
        ("Your ticket number is {num}.", lambda: random.randint(1000, 9999)),
        ("The reference code is {num}.", lambda: random.randint(100000, 999999)),
        
        # Quantities / counts (contextual)
        ("There are {num} students in the class.", lambda: random.randint(10, 50)),
        ("The book has {num} pages.", lambda: random.randint(50, 1000)),
        ("He has {num} years of experience.", lambda: random.randint(1, 40)),
        ("The team consists of {num} members.", lambda: random.randint(3, 20)),
        ("She has {num} siblings.", lambda: random.randint(0, 10)),
        
        # Versions / models
        ("This is Version {num}.", lambda: random.randint(1, 20)),
        ("The software is now at Version {num}.{sub}.", lambda: (random.randint(1, 10), random.randint(0, 9))),
        ("The model number is {num}.", lambda: random.randint(1000, 9999)),
        ("This is the {num}th edition.", lambda: random.randint(1, 20)),
        
        # Distance / measurement (contextual)
        ("The distance is {num} kilometers.", lambda: random.randint(1, 1000)),
        ("The building is {num} meters tall.", lambda: random.randint(10, 500)),
        ("The speed limit is {num} mph.", lambda: random.choice([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])),
        ("The temperature is {num} degrees.", lambda: random.randint(-10, 40)),
        
        # Price / currency (contextual)
        ("The price is ${num}.", lambda: random.randint(1, 1000)),
        ("The cost is {num} dollars.", lambda: random.randint(10, 10000)),
        ("The budget is ${num}.", lambda: random.randint(100, 100000)),
        
        # Ranks / order
        ("She finished in {num}st place.", lambda: 1),
        ("She finished in {num}nd place.", lambda: 2),
        ("She finished in {num}rd place.", lambda: 3),
        ("He ranked {num}th in the competition.", lambda: random.randint(4, 100)),
        ("This is my {num}th attempt.", lambda: random.randint(1, 20)),
        
        # Phone numbers / zip codes
        ("The zip code is {num}.", lambda: random.randint(10000, 99999)),
        ("Call {num} for more information.", lambda: random.randint(1000000, 9999999)),
        ("The area code is {num}.", lambda: random.choice([201, 212, 213, 310, 415, 510, 626, 818])),
    ]
    
    for i in range(num_samples):
        template, num_generator = random.choice(templates)
        
        # Some templates return tuples (e.g., version numbers)
        num_value = num_generator()
        if isinstance(num_value, tuple):
            text = template.format(num=num_value[0], sub=num_value[1])
        else:
            text = template.format(num=num_value)
        
        # Extract numbers from the generated text
        numbers = extract_numbers_from_text(text)
        
        data.append({
            "id": f"contextual_{i+1}",
            "text": text,
            "type": "contextual",
            "category": "semantic",
            "numbers": numbers  # numbers to track
        })
    
    return data


def generate_numerical_data(num_samples: int = 1000) -> List[Dict]:
    """
    Generate numerical/mathematical data where numbers participate
    directly in arithmetic or other computations.
    """
    data = []
    
    for i in range(num_samples):
        operation_type = random.choice([
            "addition", "subtraction", "multiplication", 
            "division", "area", "perimeter", "percentage", 
            "average", "comparison", "equation"
        ])
        
        if operation_type == "addition":
            a = random.randint(1, 1000)
            b = random.randint(1, 1000)
            templates = [
                f"The total is {a} plus {b}.",
                f"What is {a} + {b}?",
                f"Add {a} and {b}.",
                f"Calculate {a} plus {b}.",
                f"If you have {a} and add {b}, what is the sum?",
            ]
            
        elif operation_type == "subtraction":
            a = random.randint(100, 1000)
            b = random.randint(1, a)
            templates = [
                f"The difference is {a} minus {b}.",
                f"What is {a} - {b}?",
                f"Subtract {b} from {a}.",
                f"Calculate {a} minus {b}.",
                f"If you have {a} and remove {b}, what remains?",
            ]
            
        elif operation_type == "multiplication":
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            templates = [
                f"The product is {a} times {b}.",
                f"What is {a} × {b}?",
                f"Multiply {a} by {b}.",
                f"Calculate {a} times {b}.",
                f"{a} multiplied by {b} equals what?",
            ]
            
        elif operation_type == "division":
            b = random.randint(1, 50)
            a = b * random.randint(1, 100)
            templates = [
                f"The quotient is {a} divided by {b}.",
                f"What is {a} ÷ {b}?",
                f"Divide {a} by {b}.",
                f"Calculate {a} divided by {b}.",
                f"{a} split into {b} equal parts is what?",
            ]
            
        elif operation_type == "area":
            length = random.randint(5, 100)
            width = random.randint(5, 100)
            templates = [
                f"Calculate the area of a {length}x{width} rectangle.",
                f"What is the area of a rectangle with dimensions {length} by {width}?",
                f"Find the area of a {length} by {width} rectangle.",
                f"A rectangle is {length} units long and {width} units wide. What is its area?",
                f"The area of a {length}×{width} rectangle is what?",
            ]
            
        elif operation_type == "perimeter":
            length = random.randint(5, 100)
            width = random.randint(5, 100)
            templates = [
                f"Calculate the perimeter of a {length}x{width} rectangle.",
                f"What is the perimeter of a rectangle with dimensions {length} by {width}?",
                f"Find the perimeter when length is {length} and width is {width}.",
                f"A rectangle has length {length} and width {width}. What is the perimeter?",
                f"The perimeter of a {length}×{width} rectangle is what?",
            ]
            
        elif operation_type == "percentage":
            total = random.randint(100, 1000)
            percent = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
            templates = [
                f"What is {percent}% of {total}?",
                f"Calculate {percent} percent of {total}.",
                f"Find {percent}% of {total}.",
                f"{percent} percent of {total} is what?",
                f"Compute {percent}% of {total}.",
            ]
            
        elif operation_type == "average":
            nums = [random.randint(10, 100) for _ in range(random.randint(3, 5))]
            nums_str = ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"
            templates = [
                f"What is the average of {nums_str}?",
                f"Calculate the mean of {nums_str}.",
                f"Find the average: {nums_str}.",
                f"Compute the mean of {nums_str}.",
                f"The average of {nums_str} is what?",
            ]
            
        elif operation_type == "comparison":
            a = random.randint(50, 500)
            b = random.randint(50, 500)
            templates = [
                f"Which is larger: {a} or {b}?",
                f"Compare {a} and {b}.",
                f"Is {a} greater than {b}?",
                f"What is the difference between {a} and {b}?",
                f"How much more is {max(a,b)} than {min(a,b)}?",
            ]
            
        elif operation_type == "equation":
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = a + b
            templates = [
                f"Solve for x: x + {a} = {c}.",
                f"If x + {a} = {c}, what is x?",
                f"Find x when {a} + x = {c}.",
                f"What number plus {a} equals {c}?",
                f"Solve: x - {a} = {b}.",
            ]
        
        text = random.choice(templates)
        
        # Extract numbers from the generated text
        numbers = extract_numbers_from_text(text)
        
        data.append({
            "id": f"numerical_{i+1}",
            "text": text,
            "type": "numerical",
            "category": operation_type,
            "numbers": numbers  # numbers to track
        })
    
    return data


def main():
    """Main entry point: generate and save diagnostic datasets."""
    
    # Create output directory
    output_dir = "/workspace/diagnostic_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Starting Diagnostic Dataset Generator")
    print("=" * 60)
    
    # Generate contextual data
    print("\n[1/2] Generating contextual (Contextual/Semantic) data...")
    contextual_data = generate_contextual_data(num_samples=1000)
    print(f"✓ Generated {len(contextual_data)} contextual samples")
    
    # Generate numerical data
    print("\n[2/2] Generating numerical (Numerical/Mathematical) data...")
    numerical_data = generate_numerical_data(num_samples=1000)
    print(f"✓ Generated {len(numerical_data)} numerical samples")
    
    # Combine all data
    all_data = {
        "contextual": contextual_data,
        "numerical": numerical_data,
        "total_samples": len(contextual_data) + len(numerical_data)
    }
    
    # Save as a single JSON file
    output_file = os.path.join(output_dir, "diagnostic_dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved full dataset to: {output_file}")
    
    # Also save contextual and numerical datasets separately
    contextual_file = os.path.join(output_dir, "contextual_dataset.json")
    with open(contextual_file, 'w', encoding='utf-8') as f:
        json.dump(contextual_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved contextual dataset to: {contextual_file}")
    
    numerical_file = os.path.join(output_dir, "numerical_dataset.json")
    with open(numerical_file, 'w', encoding='utf-8') as f:
        json.dump(numerical_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved numerical dataset to: {numerical_file}")
    
    # Print basic statistics
    print("\n" + "=" * 60)
    print("Dataset generation finished!")
    print("=" * 60)
    print(f"Total samples: {all_data['total_samples']}")
    print(f"  - Contextual samples: {len(contextual_data)}")
    print(f"  - Numerical samples: {len(numerical_data)}")
    print("\nSample examples:")
    print("\n[Contextual samples]")
    for i in range(3):
        print(f"  {i+1}. {contextual_data[i]['text']}")
    print("\n[Numerical samples]")
    for i in range(3):
        print(f"  {i+1}. {numerical_data[i]['text']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()


