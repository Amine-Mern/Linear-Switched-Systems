from src.lpv.utils import Myu
from src.lpv.mode.modeStrategy import modeStrategy

class strategy_Myu(modeStrategy):
    """
    Class that represents the startegy where we use Myu to calculate M
    """
    def build_M(self,params):
        """
        Function that builds M

        Parameters :
        
        params: the dictionnary of parameters needed for the Myu function 

        This function mimics Myu, the documentation for the parameters
        is present in the Myu() function

        Returns :

        M : the concatenated Markov vector
        
        """

        w = params[0]
        A = params[1]
        B = params[2]
        C = params[3]
        D = params[4]
        G = params[5]
        psig = params[6]
        
        M = Myu(w,A,B,C,D,G,psig)
        
        return M
