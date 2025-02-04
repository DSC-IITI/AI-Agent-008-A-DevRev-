import fitz  # PyMuPDF
import pandas as pd

def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from each page of a PDF file using PyMuPDF and return a pandas DataFrame.

    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        pd.DataFrame: DataFrame with two columns: 'Page Number' and 'Text'.
    """
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Extract text page by page
        pages_text = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()  # Extract text using PyMuPDF
            if text.strip():  # Append only if the page contains non-empty text
                pages_text.append({'Page Number': page_num + 1, 'Text': text.strip()})
        
        # Close the document
        pdf_document.close()
        
        # Convert the data into a DataFrame
        df = pd.DataFrame(pages_text)
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()