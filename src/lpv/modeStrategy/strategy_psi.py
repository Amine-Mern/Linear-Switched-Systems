from src.lpv.utils import psi_uy_true

class strategy_psi(modeStrategy):
    """
    Class that represents the startegy where we use psi to calculate M
    """
    def build_M(self,params):
        """
        Function that builds M

        Parameters :

        params : A list of parameters needed for psi_uy_true function

        This function mimics the psi function, the documentation for the parameters is present in the psi_uy_true() function

        Returns :

        M : the concatenated Markov vector
        """
        
        w = params[0]
        A = params[1]
        B = params[2]
        C = params[3]
        D = params[4]

        M = psi_uy_true(w,A,B,C,D)

        return M
