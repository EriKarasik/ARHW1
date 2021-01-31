from numpy import transpose, identity as I, delete, zeros, \
       array, dot, pi,hstack as h,vstack as v,cos,sin, arange, \
        arctan2, log
from numpy.linalg import pinv, multi_dot, inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
Ka = 10000  # actuator stiffness
E = 7.0000e+10  # Young's modulus
G = 2.5500e+10  # shear modulus
L = 0.75
d = 0.05
A = pi * (d ** 2) / 4
Iy = pi * (d ** 4) / 64
Iz = pi * (d ** 4) / 64
Ip = Iy + Iz
def Rx(q):  return [[1, 0, 0, 0],[ 0, cos(q), -sin(q), 0],[ 0, sin(q), cos(q), 0],[0, 0, 0, 1]]
def dRx(q): return [[0, 0, 0, 0],[ 0, -sin(q),-cos(q), 0],[ 0, cos(q),-sin(q), 0],[0, 0, 0, 0]]
def Ry(q):  return [[cos(q), 0,  sin(q), 0],[0, 1, 0, 0],[-sin(q), 0,  cos(q), 0],[0, 0, 0, 1]]
def dRy(q): return [[-sin(q), 0, cos(q), 0],[0, 0, 0, 0],[-cos(q), 0, -sin(q), 0],[0, 0, 0, 0]]
def Rz(q):  return [[cos(q), -sin(q), 0, 0],[sin(q), cos(q), 0, 0],[ 0, 0, 1, 0], [0, 0, 0, 1]]
def dRz(q): return [[-sin(q), -cos(q), 0, 0],[cos(q), -sin(q), 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
def Tx(x):  return [[1, 0, 0, x],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]
def dTx():  return [[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
def Ty(y):  return [[1, 0, 0, 0],[0, 1, 0, y],[0, 0, 1, 0],[0, 0, 0, 1]]
def dTy():  return [[0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0]]
def Tz(z):  return [[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, z],[0, 0, 0, 1]]
def dTz():  return [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 1],[0, 0, 0, 0]]

T_base = [multi_dot([Ty(1),Ry(pi/2),Rz(pi)]), multi_dot([Tz(1),Rx(-pi/2)]),I(4)]


k11 = [[ E*A/L,             0,            0,       0,           0,            0],
       [     0,  12*E*Iz/L**3,            0,       0,           0,  6*E*Iz/L**2],
       [     0,             0, 12*E*Iy/L**3,       0,-6*E*Iy/L**2,            0],
       [     0,             0,            0,  G*Ip/L,           0,            0],
       [     0,             0, -6*E*Iy/L**2,       0,    4*E*Iy/L,            0],
       [     0,   6*E*Iz/L**2,            0,       0,           0,     4*E*Iz/L]]
k12 = [[-E*A/L,             0,            0,       0,           0,            0],
       [     0, -12*E*Iz/L**3,            0,       0,           0, -6*E*Iz/L**2],
       [     0,             0,-12*E*Iy/L**3,       0, 6*E*Iy/L**2,            0],
       [     0,             0,            0, -G*Ip/L,           0,            0],
       [     0,             0, -6*E*Iy/L**2,       0,    2*E*Iy/L,            0],
       [     0,   6*E*Iz/L**2,            0,       0,           0,     2*E*Iz/L]]
k22 = [[ E*A/L,             0,            0,       0,           0,            0],
       [     0,  12*E*Iz/L**3,            0,       0,           0, -6*E*Iz/L**2],
       [     0,             0, 12*E*Iy/L**3,       0, 6*E*Iy/L**2,            0],
       [     0,             0,            0,  G*Ip/L,           0,            0],
       [     0,             0,  6*E*Iy/L**2,       0,    4*E*Iy/L,            0],
       [     0,  -6*E*Iz/L**2,            0,       0,           0,     4*E*Iz/L]]
k21 = transpose(k12)
K = v([h([k11,k12]),h([transpose(k12), k22])])

le12 = [array([1,0,0,0,0,0]),array([0,1,0,0,0,0]),array([0,0,1,0,0,0])]
lr12 = [delete(I(6), (0), axis=0), delete(I(6), (1), axis=0),delete(I(6), (2), axis=0)]
lp = [[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
lr = [delete(I(6), (3), axis=0),delete(I(6), (4), axis=0),delete(I(6), (5), axis=0)]
de = [[0,-0.1,.1],[.1,0,-0.1],[-0.1,.1,0]]
De = v([h([I(3),transpose(de)]),h([zeros((3,3)),I(3)])])
def transformStiffness(p, q):
    Q = []
    for i in range(len(T_base)):
        R1 = multi_dot([T_base[i],Tz(p[i]),Rz(q[i][0])])
        R2 = multi_dot([R1,Tx(0.75),Rz(q[i][1])])
        Q1 = v([h([R1[0:3,0:3], zeros((3,3))]),h([zeros((3,3)), R1[0:3,0:3]])])
        Q2 = v([h([R2[0:3,0:3], zeros((3,3))]),h([zeros((3,3)), R2[0:3,0:3]])])
        Q.append([Q1, Q2])
    return Q
def ikLeg(T_base, p_global):
    R_base = T_base[0:3, 0:3]
    p_base = T_base[0:3, 3]
    p_local = transpose(R_base).dot(p_global - p_base)
    cos_q2 = (p_local[0]**2+p_local[1]**2-0.75**2-0.75**2)/(2*0.75*0.75)
    sin_q2 = (1-cos_q2**2)**0.5
    q2 = arctan2(sin_q2,cos_q2)
    q1 = arctan2(p_local[1],p_local[0])-arctan2(0.75*sin(q2),0.75+0.75*cos(q2))
    q3 = -(q1 + q2)
    return [q1, q2, q3]
def MSA(Q):
       Kc = []
       for i in range(3):
              wb1 = h([zeros((6,54)),I(6),zeros((6,48))])

              K11 = multi_dot([Q[i][0], k11, transpose(Q[i][0])])
              K12 = multi_dot([Q[i][0], k12, transpose(Q[i][0])])
              K21 = multi_dot([Q[i][0], k21, transpose(Q[i][0])])
              K22 = multi_dot([Q[i][0], k22, transpose(Q[i][0])])

              w45 = v([h([zeros((6,18)),-I(6),zeros((6,48)),K11,K12,zeros((6,24))]),
                       h([zeros((6,24)),-I(6),zeros((6,42)),K21,K22,zeros((6,24))])])

              K11 = multi_dot([Q[i][1], k11, transpose(Q[i][1])])
              K12 = multi_dot([Q[i][1], k12, transpose(Q[i][1])])
              K21 = multi_dot([Q[i][1], k21, transpose(Q[i][1])])
              K22 = multi_dot([Q[i][1], k22, transpose(Q[i][1])])

              w67 = v([h([zeros((6,30)),-I(6),zeros((6,48)),K11,K12,zeros((6,12))]),
                       h([zeros((6,36)), -I(6), zeros((6,42)),K21,K22, zeros((6,12))])])

              w8e = v([h([zeros((6,96)),De,-I(6)]),
                       h([zeros((6,42)),I(6),transpose(De),zeros((6,54))])])

              w23 = v([h([zeros((6,60)),I(6),-I(6),zeros((6,36))]),
                       h([zeros((6,6)),I(6),I(6),zeros((6,90))])])

              w12 = v([h([zeros((5,54)),lr12[i],-lr12[i],zeros((5,42))]),
                       h([I(6),I(6),zeros((6,96))]),
                       h([le12[i],zeros((48)),Ka*le12[i],-Ka*le12[i],zeros((42))])])

              w34 = v([h([zeros((5,66)),lr[i],-lr[i],zeros((5,30))]),
                       h([zeros((5,12)),lr[i],lr[i],zeros((5,84))]),
                       h([zeros((12)),lp[i],zeros((90))]),
                       h([zeros((18)),lp[i],zeros((84))])])


              w56 = v([h([zeros((5,78)),lr[i],-lr[i],zeros((5,18))]),
                       h([zeros((5,24)),lr[i],lr[i],zeros((5,72))]),
                       h([zeros((24)),lp[i],zeros((78))]),
                       h([zeros((30)),lp[i],zeros((72))])])

              w78 = v([h([zeros((5,90)),lr[i],-lr[i],zeros((5,6))]),
                       h([zeros((5,36)),lr[i],lr[i],zeros((5,60))]),
                       h([zeros((36)),lp[i],zeros((66))]),
                       h([zeros((42)),lp[i],zeros((60))])])

              wagr = h([zeros((6,48)),-I(6),zeros((6,54))])
              # Aggregated matrix
              agg = v([wb1,w45,w67,w8e,w23,w12,w34,w56,w78,wagr])

              A = agg[0:102, 0:102]
              B = agg[0:102, 102:108]
              C = agg[102:108, 0:102]
              D = agg[102:108, 102:108]

              K_leg = D - multi_dot([C, pinv(A), B])
              Kc.append(K_leg)
       return Kc[0] + Kc[1] + Kc[2]

xScatter, yScatter,zScatter,dScatter = [],[],[],[]

F = array([[1], [0], [0], [0], [0], [0]], dtype=float)
start = 0.01
step = 0.1
count = 0
for z in arange(0.1, 1.1, 0.1):
       for x in arange(0.1, 1.1, 0.1):
              for y in arange(0.1, 1.1, 0.1):
                     q = []
                     for leg in range(len(T_base)):
                            q.append(ikLeg(T_base[leg], array([x, y, z])))

                     Q = transformStiffness([x, y, z], q)

                     Kc = MSA(Q)

                     dt = inv(Kc).dot(F)
                     deflection = (dt[0]**2+dt[1]**2+dt[2]**2)**0.5
                     xScatter.append(x)
                     yScatter.append(y)
                     zScatter.append(z)
                     dScatter.append(deflection[0])
       #plt.close()
cmap = plt.cm.get_cmap('RdGy_r', 12)

for i in range(len(dScatter)): dScatter[i] = log(dScatter[i])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
cmap = ax.scatter3D(xScatter, yScatter, zScatter, c=dScatter, cmap=cmap, s=60)
plt.colorbar(cmap)
plt.show()