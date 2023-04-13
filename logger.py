import time


class Logger:
    def __init__(self, logger_name: str):
        self.logger_name = logger_name

    def log(self, content: str):
        with open(self.logger_name + '.log', 'a') as f:
            f.write('@' + str(time.asctime(time.localtime(time.time()))) + ': ' + content + '\n')
