from abc import ABC, abstractmethod

class modeStrategy(ABC) :
    """
    Abstract class (used like an interface) representing the mode with what TrueHoKalmanBase will construct M
    """
    @abstractmethod
    def build_M(self):
        pass
    
