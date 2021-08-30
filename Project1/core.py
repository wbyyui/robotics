from __future__ import print_function

import numpy as np
#import scipy as sp
#from scipy import linalg


def near_zero(z):
    """Determines whether a scalar is small enough to be treated as zero

    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise

    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6


def normalize(V):
    """Normalizes a vector

    :param V: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    if near_zero(np.linalg.norm(V)):
        return V
    return V / np.linalg.norm(V)


'''
RIGID-BODY MOTIONS
'''


def rot_inv(R):
    """Inverts a rotation matrix

    :param R: A rotation matrix
    :return: The inverse of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return R.T


def vec_to_so3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return np.array([[ 0, -omg[2],  omg[1]],
                  [ omg[2],  0, -omg[0]],
                  [-omg[1],  omg[0],  0]])


#omg = np.array([1, 2, 3])
#print(vec_to_so3(omg))

def so3_to_vec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])


#so3mat = np.array([[ 0, -3,  2],[ 3,  0, -1],[-2,  1,  0]])
#print(so3_to_vec(so3mat))

def axis_ang3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return (normalize(expc3), np.linalg.norm(expc3))


#expc3 = np.array([1, 2, 3])
#print(axis_ang3(expc3))

def matrix_exp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    theta = axis_ang3(so3_to_vec(so3mat))[1]
    if near_zero(theta): return np.identity(3)
    return (np.identity(3) + (np.sin(theta)/theta)*so3mat + (2*np.sin(theta/2)*np.sin(theta/2)/theta/theta)*(np.matmul(so3mat,so3mat)))

#so3mat = np.array([[ 0, 0,  0], [ 0,  0, -1.57], [0,  1.57,  0]])
#print(matrix_exp3(so3mat))


def matrix_log3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    theta = np.arccos((np.trace(R)-1)/2)
    if near_zero(np.sin(theta)):
        if near_zero(np.cos(theta)-1): return np.zeros((3,3))
        else:
            for i in range(0,3):
                if not (near_zero(R[i][i]+1)):
                    w = (1/np.sqrt(2*(R[i][i]+1))) * np.array([(int)(0==i) + R[0][i], (int)(1==i) + R[1][i], (int)(2==i) + R[2][i]])
                    return vec_to_so3(w) * np.pi
    return (theta/(2*np.sin(theta))) *(R - R.T)

#R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#print(matrix_log3(R))   
#print(scipy.linalg.logm(R))

def rp_to_trans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return np.array([
        [R[0][0], R[0][1], R[0][2], p[0]],
        [R[1][0], R[1][1], R[1][2], p[1]],
        [R[2][0], R[2][1], R[2][2], p[2]],
        [0,0,0,1]
    ])


def trans_to_rp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return (np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]]
    ]), np.array([T[0][3],T[1][3],T[2][3]]))

#T = np.array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1,  0, 3], [0, 0,  0, 1]])
#print(trans_to_rp(T)[0])


def trans_inv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return rp_to_trans(trans_to_rp(T)[0].T,np.matmul(((-1)*trans_to_rp(T)[0].T),trans_to_rp(T)[1]))

#T = np.array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1,  0, 3], [0, 0,  0, 1]])
#print(trans_inv(T))


def vec_to_se3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3

    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    R = vec_to_so3(np.array(V[0:3]))
    se3mat = rp_to_trans(R, np.array(V[3:6]))
    se3mat[3][3] = 0
    return se3mat

#V = np.array([1, 2, 3, 4, 5, 6])
#print(vec_to_se3(V))


def se3_to_vec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return np.array([se3mat[2][1],se3mat[0][2],se3mat[1][0],se3mat[0][3],se3mat[1][3],se3mat[2][3]])


#se3mat = np.array([[ 0, -3,  2, 4], [ 3,  0, -1, 5], [-2,  1,  0, 6], [ 0,  0,  0, 0]])
#print(se3_to_vec(se3mat))   


def adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    (R,p)=trans_to_rp (T)
    #print(R)#print(p)
    return np.vstack((np.hstack((R,np.zeros((3,3),dtype=int))),np.hstack((np.matmul(vec_to_so3(p),R),R))))

#T = np.array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1,  0, 3], [0, 0,  0, 1]])
#print(adjoint(T))


def screw_to_axis(q, s, h):
    """Takes a parametric description of a screw axis and converts it to a
    normalized screw axis

    :param q: A point lying on the screw axis
    :param s: A unit vector in the direction of the screw axis
    :param h: The pitch of the screw axis
    :return: A normalized screw axis described by the inputs

    Example Input:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
    Output:
        np.array([0, 0, 1, 0, -3, 2])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    return np.hstack((s,np.cross(-1*s,q)+h*s))

#q = np.array([3, 0, 0])
#s = np.array([0, 0, 1])
#h = 2
#print(screw_to_axis(q, s, h))

def axis_ang6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S

    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    theta = np.linalg.norm(expc6[0:3])
    if near_zero(theta): theta = np.linalg.norm(expc6[3:6])
    return (expc6/theta,theta)

#expc6 = np.array([1, 0, 1, 1, 2, 3])
#print(axis_ang6(expc6))


def matrix_exp6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates

    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat

    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''

    #print(sp.linalg.expm(T))

    w = se3mat[0:3, 0:3]
    theta = axis_ang3(so3_to_vec(w))[1]
    if near_zero(theta): return (rp_to_trans(np.identity(3),se3mat[0:3, 3]))

    w = w / theta
    R = matrix_exp3(se3mat[0:3, 0:3])
    G = theta*np.identity(3) + (1-np.cos(theta))*w + (theta - np.sin(theta)) * np.matmul(w,w)
    return rp_to_trans(R,np.matmul(G,se3mat[0:3, 3]/theta))
    
#se3mat = np.array([[0,0,0,0],[0,0,-1.57079632,2.35619449],[0,1.57079632,0, 2.35619449],[0,0,0,0]])
#print(matrix_exp6(se3mat))

def matrix_log6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix

    :param R: A matrix in SE3
    :return: The matrix logarithm of R

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''

    #print(sp.linalg.logm(T))

    R,p = trans_to_rp(T)
    theta = np.arccos((np.trace(R)-1)/2)
    omega = matrix_log3(R)

    if near_zero(np.linalg.norm(omega)):
        return vec_to_se3(np.hstack((so3_to_vec(omega),p)))
    
    InvG = (np.identity(3) - omega/2 + ((1/theta - 1/(2*np.tan(theta/2))))/theta*np.matmul(omega,omega))
    #Here the InvG given by PPT may be with some typos

    return vec_to_se3(np.hstack((so3_to_vec(omega),np.matmul(InvG, p))))

#T = np.array([[1, 0,  0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]])
#print(matrix_log6(T))



'''
*** FORWARD KINEMATICS ***
'''


def FK_in_body(M, Blist, thetalist):
    """Computes forward kinematics in the body frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             in body frame

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''
    Blist = Blist.T
    n = len(thetalist)
    for i in range(n):
        M = np.matmul(M,matrix_exp6(vec_to_se3(Blist[i]*thetalist[i])))
    return M

def FK_in_space(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot

    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             in space frame

    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """

    '''-----------------------'''
    '''----Your Code HERE:----'''
    '''-----------------------'''

    Slist = Slist.T
    n = len(thetalist)
    for i in range(n-1,-1,-1):
        M = np.matmul(matrix_exp6(vec_to_se3(Slist[i]*thetalist[i])),M)
    return M

#M = np.array([[-1, 0,  0, 0], [ 0, 1,  0, 6], [ 0, 0, -1, 2],[ 0, 0,  0, 1]])
#Blist = np.array([[0, 0, -1, 2, 0,   0],[0, 0,  0, 0, 1,   0],[0, 0,  1, 0, 0, 0.1]]).T
#Slist = np.array([[0, 0,  1,  4, 0,    0],[0, 0,  0,  0, 1,    0],[0, 0, -1, -6, 0, -0.1]]).T
#thetalist = np.array([np.pi / 2.0, 3, np.pi])
#print(FK_in_body(M, Blist, thetalist))
#print(FK_in_space(M, Slist, thetalist))