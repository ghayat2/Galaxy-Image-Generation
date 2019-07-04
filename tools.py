import os, sys
import datetime, time
import patoolib

class Logger(object):  # logger to log output to both terminal and file
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_dir, "output"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()
        self.terminal.flush()
        pass    

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")
    
def create_zip_code_files(output_file, submission_files):
    patoolib.create_archive(output_file, submission_files)
