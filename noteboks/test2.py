from src.exception import CustomException
import sys

def test_custom_exception():
    try:
        x = 10 / 0
    except Exception as e:
        error = CustomException("Something went wrong!", sys)
        print("Custom Error:", error)
    
test_custom_exception()