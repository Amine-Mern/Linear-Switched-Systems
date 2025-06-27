from src.lpv.utils import psi_uy_true

class strategy_psi(modeStrategy):
    """
    Class that represents the startegy where we use psi to calculate M
    """
    def build_M(self,w,A,B,C,D):
        """
        Function that builds M

        This function mimics the psi function, the documentation for the parameters is present in the psi_uy_true() function

        Returns :

        M : the concatenated Markov vector
        """

        M = psi_uy_true(w,A,B,C,D)
        return M
