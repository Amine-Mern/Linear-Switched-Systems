"""
NEEDS TO BE CONVERTED TO OOP
"""

import numpy as np
import scipy

def isEquivalent(D,n1,n2,m,p,Anum1,Bnum1,Cnum1,x01, Anum2,Bnum2,Cnum2,x02) :
    """
    Used to test input-ouput equality between two LPV systems,
    comparing their Markov parameters to the nth order
    Args :
        D :  Input Output
        n1 :
        n2 :
        m :
        p :
        Anum1 :
        Bnum1 :
        Cnum1 :
        x01 :
        Anum2 :
        Bnum2 :
        Cnum2 :
        x02 :
    
    Returns :
        b : Boolean
    """
    epsilon = 10**(-5)
    R = [x01,Bnum1,x02,Bnum2]
    compC = [Cnum1,-Cnum2]
     