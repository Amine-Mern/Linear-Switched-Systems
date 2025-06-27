from src.lpv.utils import Myu

class strategy_Myu(modeStrategy):
    """
    Class that represents the startegy where we use Myu to calculate M
    """
    def build_M(self,w,A,B,C,D,G,psig): #change to build_M(self,params)
        """
        Function that builds M

        This function mimics Myu, the documentation for the parameters
        is present in the Myu() function

        Returns :

        M : the concatenated Markov vector
        
        """

        sig_j = w[0]
        v_j = w[1]
        sig = w[2]
        u_i = w[3]
        M = Myu(sig_j,v_j,sig,u_i,A,B,C,D,G,psig)
        return M
