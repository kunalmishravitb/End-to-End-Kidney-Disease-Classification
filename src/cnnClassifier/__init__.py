import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs" # directory to save logs
log_filepath = os.path.join(log_dir,"running_logs.log") # file to save logs
os.makedirs(log_dir, exist_ok=True) # create log directory if it doesn't exist


logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers=[
        logging.FileHandler(log_filepath), # log to a file
        logging.StreamHandler(sys.stdout) # log to console (terminal)
    ]
)

logger = logging.getLogger("cnnClassifierLogger") # logger object

