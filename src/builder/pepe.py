#!/usr/bin/env python3


class Pepe:
    def __init__(self):
        print("Pepe Init")
        self.name = "Pepe"
    def __call__(self):
        print("Pepe Call")
        return self.name

pepe = Pepe()  # This prints "Pepe Init"
pepe()  # This prints "Pepe Call"
