import os
import random
import time
import sys
from colorama import init, Fore, Back, Style

init(autoreset=True)

class Matrix:
    def __init__(self, width=80, height=24):
        self.width = width
        self.height = height
        self.columns = [0] * self.width

    def rain(self):
        while True:
            for i in range(self.width):
                if self.columns[i] == 0:
                    if random.random() < 0.02:
                        self.columns[i] = random.randint(1, self.height)
                elif self.columns[i] > 0:
                    if random.random() < 0.05:
                        self.columns[i] = 0
                    else:
                        self.columns[i] += 1

            self.display()
            time.sleep(0.1)

    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in range(self.height):
            line = ''
            for col in range(self.width):
                if self.columns[col] > row:
                    line += Fore.GREEN + chr(random.randint(33, 126)) + ' '
                else:
                    line += '  '
            print(line)

if __name__ == "__main__":
    width, height = os.get_terminal_size()
    matrix = Matrix(width // 2, height)
    try:
        matrix.rain()
    except KeyboardInterrupt:
        sys.exit(0)
