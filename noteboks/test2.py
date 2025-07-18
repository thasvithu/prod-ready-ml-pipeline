import logging

from src.exception import CustomException
import sys

def test_custom_exception():
    try:
        x = 10 / 0
    except Exception as e:
        logging.info("Division by zero error occurred.")
        error = CustomException("Something went wrong!", sys)
        print("Custom Error:", error)
    
test_custom_exception()