import sys
from logger import logging

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() #Information about where the error has occurred

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno

    error_message= f'Error occured in {file_name}. \n Line number:{line_no}. \n Error message: {str(error)}'

    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)


    def __str__(self):
        '''
        Whenever we print the exception we'll see the error message
        '''
        return self.error_message