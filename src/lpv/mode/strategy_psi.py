from src.lpv.utils import psi_uy_true
from src.lpv.mode.modeStrategy import modeStrategy

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

    def search_hak_row_dim(self,nu,ny):
        """
        Method that returns the dimension needed for the third sub-hankel matrix depends on the strategy used (Myu or Psi)
        """
        return nu

    def return_tuple(self,Bi,Gi,D,F):
        return (Bi,D)

    def select_sep(self,Bi,Gi):
        return Bi 
