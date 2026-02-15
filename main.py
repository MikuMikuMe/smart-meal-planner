Creating a comprehensive smart meal planner with machine learning capabilities involves several components: dietary preferences input, nutritional goals definition, ingredient availability, and a machine learning model for optimizing meal plans. Below is a simplified Python program demonstrating these components using mock data and a basic recommendation system. In a real-world application, you would connect this with detailed nutritional databases, user input systems, and a more sophisticated recommendation engine. 

This program assumes installation of several libraries such as `pandas`, `numpy`, `sklearn` for machine learning, and `random` for simulating ingredient availability.

```python
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Dict

# Mock database of meals with nutritional info (calories, protein, carbs, fat)
MEAL_DATABASE = pd.DataFrame([
    {'meal': 'Chicken Salad', 'calories': 400, 'protein': 35, 'carbs': 20, 'fat': 20},
    {'meal': 'Beef Stew', 'calories': 900, 'protein': 60, 'carbs': 40, 'fat': 50},
    {'meal': 'Vegetable Stir Fry', 'calories': 350, 'protein': 10, 'carbs': 50, 'fat': 10},
    {'meal': 'Tofu Curry', 'calories': 450, 'protein': 20, 'carbs': 60, 'fat': 15},
    {'meal': 'Grilled Salmon', 'calories': 600, 'protein': 45, 'carbs': 10, 'fat': 35},
])

# User input: dietary preferences, nutritional goals, ingredient availability
user_preferences = {
    'diet_type': 'balanced',  # Options: 'balanced', 'high-protein', 'low-carb', etc.
    'calories': 2000,
    'protein': 100,
    'carbs': 250,
    'fat': 70,
}

# Simulate available ingredients
available_ingredients = ["chicken", "lettuce", "tomato", "beef", "carrot", "peppers", "tofu", "spinach"]


def check_ingredient_availability(meal: str, ingredients: List[str]):
    """
    Simulates check if ingredients of a meal are available.
    """
    # Mock check - random availability check
    return random.choice([True, False])


def error_handling():
    """
    Function to handle errors, such as data inconsistencies or lack of available ingredients.
    """
    try:
        # Check if meal ingredients are available
        for index, row in MEAL_DATABASE.iterrows():
            if not check_ingredient_availability(row['meal'], available_ingredients):
                print(f"Warning: Some ingredients for {row['meal']} are not available.")
    except Exception as e:
        print(f"An error occurred during ingredient check: {e}")


def personalize_meal_plan(user_prefs: Dict, meal_data: pd.DataFrame):
    """
    Generate a personalized meal plan based on user preferences using k-means clustering.
    """
    # Normalize the data
    scaler = StandardScaler()
    features = meal_data[['calories', 'protein', 'carbs', 'fat']]
    normalized_features = scaler.fit_transform(features)

    # Apply k-means clustering to find meal clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(normalized_features)
    meal_data['cluster'] = kmeans.labels_

    # Recommend meals based on user's dietary type (mock logic)
    target_cluster = 0  # Default cluster for 'balanced'
    if user_prefs['diet_type'] == 'high-protein':
        target_cluster = 1
    elif user_prefs['diet_type'] == 'low-carb':
        target_cluster = 2

    recommended_meals = meal_data[meal_data['cluster'] == target_cluster]

    # Filter based on nutrient goals
    plan = recommended_meals[
        (recommended_meals['calories'] <= user_prefs['calories']) &
        (recommended_meals['protein'] <= user_prefs['protein']) &
        (recommended_meals['carbs'] <= user_prefs['carbs']) &
        (recommended_meals['fat'] <= user_prefs['fat'])
    ]

    return plan


def main():
    print("Welcome to the Smart Meal Planner!")
    
    # Handle potential errors when accessing data
    error_handling()

    # Generate meal plan
    meal_plan = personalize_meal_plan(user_preferences, MEAL_DATABASE)
    
    if not meal_plan.empty:
        print("Here is your personalized meal plan:")
        print(meal_plan[['meal', 'calories', 'protein', 'carbs', 'fat']])
    else:
        print("Sorry, no meals matched your criteria.")

if __name__ == "__main__":
    main()
```

### Explanation:
1. **Mock Database & User Preferences:** 
   - Contains pre-defined meals with nutritional values.
   - User preferences include dietary type (e.g., balanced, high-protein) and nutritional goals.

2. **Ingredient Availability Check:** 
   - Random function simulates ingredient availability.
   - Prints warnings if any ingredients for meals are not available.

3. **Error Handling:** 
   - Simple handling with try-except for checking the meal database and other operations.

4. **Meal Plan Personalization Using Machine Learning (K-Means Clustering):** 
   - Normalizes meal data for clustering.
   - Groups meals into clusters using K-Means.
   - Recommends meals based on the user's diet type and goals (cluster logic demonstrates target group).

5. **Executable Program:**
   - Runs the planner and displays a meal plan if an appropriate match is found.

This program is a basic representation, and real-world applications require expanded datasets and robust user interaction systems, possibly involving web frameworks for handling user inputs dynamically.