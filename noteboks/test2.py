from src.exception import CustomException
import sys

def test_custom_exception():
    try:
        # Intentionally causing a division by zero error
        x = 10 / 0
    except Exception as e:
        # Raising CustomException with the caught exception details
        raise CustomException("Something went wrong!", sys) from e
    
test_custom_exception()