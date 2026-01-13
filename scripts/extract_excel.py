import pandas as pd
import sys

def extract_excel_data(file_path):
    try:
        # Load all sheets from the excel file
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        for sheet_name in sheet_names:
            print(f"\n{'='*20} SHEET: {sheet_name} {'='*20}\n")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Output the content as a string
            print(df.to_string())
            
            print(f"\n--- DETAILED ROW DATA (Sheet: {sheet_name}) ---\n")
            for index, row in df.iterrows():
                print(f"Row {index}:")
                for col in df.columns:
                    print(f"  {col}: {row[col]}")
                print("-" * 20)
            
    except Exception as e:
        print(f"Error reading excel file: {e}")
            
    except Exception as e:
        print(f"Error reading excel file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_excel_data(sys.argv[1])
    else:
        print("Usage: python extract_excel.py <path_to_xlsx>")
