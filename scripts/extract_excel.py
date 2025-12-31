import pandas as pd
import sys

def extract_excel_data(file_path):
    try:
        # Load the excel file
        # Assuming the first sheet contains the data
        df = pd.read_excel(file_path)
        
        # Output the content as a string
        print(df.to_string())
        
        # Also print detailed info about columns if needed, but to_string usually gives a good overview for small files
        # If the file is complex, we might need to iterate rows
        print("\n--- DETAILED ROW DATA ---\n")
        for index, row in df.iterrows():
            print(f"Row {index}:")
            for col in df.columns:
                print(f"  {col}: {row[col]}")
            print("-" * 20)
            
    except Exception as e:
        print(f"Error reading excel file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_excel_data(sys.argv[1])
    else:
        print("Usage: python extract_excel.py <path_to_xlsx>")
