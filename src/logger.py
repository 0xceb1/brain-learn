import logging
import warnings
import os
import datetime
import getpass
from typing import Optional
from pathlib import Path


class Logger:
    def __init__(
        self,
        job_name: str,
        console_log: bool = False,
        file_log: bool = True,
        logs_directory: Optional[str] = None,
        incremental_run_number: bool = True,
    ):

        self.logger = logging.getLogger(job_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if console_log:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s|%(message)s'))
            self.logger.addHandler(console_handler)

        self.file_name = None
        if file_log and logs_directory:
            try:
                self.job_name = job_name
                dt = datetime.datetime.now()

                log_directory = Path(logs_directory) / job_name / dt.strftime('%Y%m')
                log_directory.mkdir(parents=True, exist_ok=True, mode=0o775)

                if not os.access(log_directory, os.W_OK):
                    warnings.warn(
                        f'{log_directory} is read-only, switching to console log only',
                        UserWarning,
                        stacklevel=2,
                    )
                    if not console_log:
                        console_handler = logging.StreamHandler()
                        console_handler.setFormatter(
                            logging.Formatter('%(asctime)s|%(message)s')
                        )
                        self.logger.addHandler(console_handler)
                else:
                    # Generate file name with run number if needed
                    self.job_id = (
                        f'{job_name}_{dt.strftime("%Y%m%d")}_{getpass.getuser()}'
                    )
                    run_number = 0

                    if incremental_run_number:
                        pattern = f'{self.job_id}_*.log'
                        existing_logs = list(log_directory.glob(pattern))
                        if existing_logs:
                            # Extract run numbers from filenames and find the maximum
                            run_numbers = []
                            for log_file in existing_logs:
                                try:
                                    run_num = int(log_file.stem.split('_')[-1])
                                    run_numbers.append(run_num)
                                except ValueError, IndexError:
                                    continue
                            if run_numbers:
                                run_number = max(run_numbers) + 1

                    self.file_name = log_directory / f'{self.job_id}_{run_number}.log'
                    file_handler = logging.FileHandler(self.file_name, 'a')
                    file_handler.setFormatter(
                        logging.Formatter(
                            '%(asctime)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
                        )
                    )
                    self.logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                warnings.warn(
                    f'Failed to set up file logging: {e}. Falling back to console logging.',
                    UserWarning,
                    stacklevel=2,
                )
                if not console_log:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(
                        logging.Formatter('%(asctime)s|%(message)s')
                    )
                    self.logger.addHandler(console_handler)

    def log(self, text: str) -> None:
        self.logger.info(text)

    def warning(self, text: str) -> None:
        self.logger.warning(f'[WARNING:]{text}')

    def error(self, text: str) -> None:
        self.logger.error(f'[ERROR:]{text}')

    def critical(self, text: str) -> None:
        self.logger.critical(f'[CRITICAL:]{text}')
