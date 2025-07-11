from abc import ABC, abstractmethod

class modeStrategy(ABC) :
    """
    Abstract class (used like an interface) representing the mode with what TrueHoKalmanBase will construct M
    """
    @abstractmethod
    def build_M(self,params):
        pass
    
    @abstractmethod
    def search_hak_row_dim(self):
        pass
