import os, sys
import json

class ConfigParser():
    def __init__(self, configPath:str):
        self.configFilePath = configPath
        self.load()
    
    def load(self,):
        with open(self.configFilePath,'r') as file:
            self.config = json.load(file)
    
if __name__ == "__main__":
    configFile = "config.json"
    parser = ConfigParser(configPath=configFile)
    print(parser.config)