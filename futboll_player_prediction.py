# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:55:37 2025

@author: daisy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import customtkinter as ctk

# Load dataset (replace 'cleaned_data.csv' with your file path)
file_path = r"C:\Users\daisy\Downloads\remaining_data.csv"  # Kullanıcının belirttiği doğru dosya yolu
data = pd.read_csv(file_path)
data['Value_M'] = data['Value'] / 1_000_000

# Select features and target variable
features = [
    'Age', 'Overall', 'Potential', 'Dribbling', 'Finishing', 'ShortPassing',
    'LongPassing', 'Composure', 'Vision', 'Acceleration', 'SprintSpeed',
    'International Reputation', 'BMI'
]
target = 'Value_M'

# Split data into features (X) and target (y)
X = data[features]
y = data[target]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into train, test, and validation sets (60%-20%-20% split)
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# GUI Implementation
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Fixed ranges for player attributes
fixed_ranges = {
    'Age': (16, 45),
    'Overall': (0, 100),
    'Potential': (0, 100),
    'Dribbling': (0, 100),
    'Finishing': (0, 100),
    'ShortPassing': (0, 100),
    'LongPassing': (0, 100),
    'Composure': (0, 100),
    'Vision': (0, 100),
    'Acceleration': (0, 100),
    'SprintSpeed': (0, 100),
    'International Reputation': (1, 5),
    'BMI': (16, 35)
}

# Main Application
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Football Player Value Prediction")
        self.geometry("800x600")

        self.show_welcome_page()

    def show_welcome_page(self):
        self.clear_frame()
        
        ctk.CTkLabel(self, text="Welcome to Player Value Prediction", font=("Helvetica", 24)).pack(pady=40)
        ctk.CTkButton(self, text="Continue", command=self.show_selection_page, fg_color="#FFB6C1").pack(pady=20)

    def show_selection_page(self):
        self.clear_frame()
        
        ctk.CTkLabel(self, text="What would you like to do?", font=("Helvetica", 20)).pack(pady=20)
        ctk.CTkButton(self, text="Register New Player", command=self.show_new_player_page, fg_color="#FF69B4").pack(pady=10)
        ctk.CTkButton(self, text="View Existing Players", command=self.show_existing_players, fg_color="#FF69B4").pack(pady=10)

    def show_new_player_page(self):
        self.clear_frame()

        entries = {}
        ctk.CTkLabel(self, text="Enter Player Details", font=("Helvetica", 20)).grid(row=0, column=0, columnspan=2, pady=10)

        for i, feature in enumerate(features):
            min_val, max_val = fixed_ranges[feature]
            ctk.CTkLabel(self, text=f"{feature} ({min_val}-{max_val}):", anchor="e").grid(row=i + 1, column=0, sticky="e", padx=10, pady=5)
            entry = ctk.CTkEntry(self)
            entry.grid(row=i + 1, column=1, padx=10, pady=5)
            entries[feature] = entry

        def predict_player_value():
            try:
                new_player = []
                for feature in features:
                    value = float(entries[feature].get())
                    min_val, max_val = fixed_ranges[feature]
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{feature} must be between {min_val} and {max_val}.")
                    new_player.append(value)

                # Normalize the new player's features using the same scaler
                new_player_df = pd.DataFrame([new_player], columns=features)
                new_player_normalized = scaler.transform(new_player_df)

                # Predict the value for the new player
                predicted_value = rf_model.predict(new_player_normalized)[0]
                result_label.configure(text=f"Predicted Value: {predicted_value:.2f}M", fg_color="#FFB6C1")

            except ValueError as e:
                result_label.configure(text=str(e), fg_color="#e74c3c")
            except Exception as e:
                result_label.configure(text=f"Error: {str(e)}", fg_color="#e74c3c")

        # Add predict button
        predict_button = ctk.CTkButton(self, text="Predict", command=predict_player_value, fg_color="#FFB6C1")
        predict_button.grid(row=len(features) + 1, column=0, columnspan=2, pady=20)

        # Add result label
        result_label = ctk.CTkLabel(self, text="", font=("Helvetica", 16))
        result_label.grid(row=len(features) + 2, column=0, columnspan=2, pady=10)

    def show_existing_players(self):
        self.clear_frame()

        ctk.CTkLabel(self, text="Existing Players", font=("Helvetica", 20)).pack(pady=20)

        search_frame = ctk.CTkFrame(self)
        search_frame.pack(pady=10)

        search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search by Name")
        search_entry.grid(row=0, column=0, padx=10)

        def search_player():
            query = search_entry.get().strip().lower()
            results = data[data['Name'].str.lower().str.contains(query, na=False)]

            for widget in result_frame.winfo_children():
                widget.destroy()

            if not results.empty:
                for i, col in enumerate(['Name'] + features + ["Value_M"]):
                    ctk.CTkLabel(result_frame, text=col, font=("Helvetica", 12, "bold")).grid(row=0, column=i, padx=5, pady=5)

                for row_index, (_, row) in enumerate(results.iterrows()):
                    for col_index, col in enumerate(['Name'] + features + ["Value_M"]):
                        ctk.CTkLabel(result_frame, text=f"{row[col]}").grid(row=row_index + 1, column=col_index, padx=5, pady=5)
            else:
                ctk.CTkLabel(result_frame, text="No results found.", font=("Helvetica", 16), fg_color="#e74c3c").pack(pady=10)
                ctk.CTkButton(result_frame, text="Try Again", command=self.show_existing_players, fg_color="#FFB6C1").pack(pady=10)

        ctk.CTkButton(search_frame, text="Search", command=search_player, fg_color="#FFB6C1").grid(row=0, column=1, padx=10)

        result_frame = ctk.CTkFrame(self)
        result_frame.pack(pady=20)

        ctk.CTkButton(self, text="Back", command=self.show_selection_page, fg_color="#90CAF9").pack(pady=20)

    def clear_frame(self):
        for widget in self.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
