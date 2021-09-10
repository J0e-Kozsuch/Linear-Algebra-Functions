import numpy as np
import timeit as t
import scipy as sci
import time
from sympy import *

A=np.identity(4)*3
ones=np.diag([3,3,3],-1)
twos=np.diag([3,3,3],1)
A=A-ones



def qr(A):
    shape=A.shape
    R=np.zeros((shape[1],shape[1]))
    Q=np.zeros(shape)
    for j in range(0,shape[1]): 
        qNew=A[:,j]
        for k in range(0,j):
            qProj=Q[:,k]
            qNew=qNew-np.dot(qNew,qProj)*qProj
        qNew=qNew/np.linalg.norm(qNew)
        Q[:,j]=qNew
        R[j,j:]=qNew@A[:,j:]
             
    return(Q,R)



def rr(A):
    U=np.array(A,dtype=float)
    rows=U.shape[0]
    pivotNum=0
    columns=U.shape[1]
    permNum=0
    for j in range(columns):                #iterate through columns
        counter=pivotNum
        while counter<rows:
            if abs(U[counter,j])<10**(-10):     #if pivot is 0 move to next row
                counter+=1
                continue
            if counter>j:           #if pivot is not in (j,j) location, switch rows
                placeholder=U[pivotNum,:].copy()
                U[pivotNum,:]=U[counter,:].copy()
                U[counter,:]=placeholder
                permNum+=1
            pivot=U[pivotNum,j]
            for i in range(pivotNum+1,rows):           #iterate through rows beneath pivot to delete entry
                if abs(U[i,j])<10**(-10):
                    continue
                multiplier=-U[i,j]/pivot        #multiple pivot row by amount
                U[i,:]=U[i,:]+U[pivotNum,:]*multiplier     #subtract one row from the other
            pivotNum+=1
            break
    U=np.round(U,10)
    return U,permNum



def lu(A):      #determinent sign should reflect the number of permutations
    U=np.array(A,dtype=float)
    rows=U.shape[0]
    columns=U.shape[1]
    L=np.identity(rows)                    #l multiply each permutation and elimination as they occur
    P=np.identity(rows)
    permNum=0
    for j in range(columns):                #iterate through columns
        counter=j
        E="a"                       #"a" is placeholder to draw error
        while counter<rows:
            if abs(U[counter,j])<10**(-10):     #if pivot is 0 move to next row
                counter+=1
                continue
            if counter>j:           #if pivot is not in (j,j) location, switch rows
                placeholder=U[j,:].copy()
                U[j,:]=U[counter,:]
                U[counter,:]=placeholder             
                perm=np.identity(rows)          #create permutation matrix to switch rows
                perm[counter,j]=1
                perm[j,counter]=1
                perm[j,j]=0
                perm[counter,counter]=0
                P=perm@P                        #multiple P by perm
                permNum+=1
            pivot=U[j,j]
            for i in range(j+1,rows):           #iterate through rows beneath pivot to delete entry
                if abs(U[i,j])<10**(-10):
                    continue
                multiplier=-U[i,j]/pivot        #multiple pivot row by amount

                U[i,:]=U[i,:]+U[j,:]*multiplier     #subtract one row from the other
                try:                            #calculate E^-1 and multiply to L
                    E[i,j]=-multiplier
                except:
                    E=np.identity(rows)
                    E[i,j]=-multiplier

            try:
                L=L@E
            except:
                pass
            break
    U=np.round(U,10)
    return P,L,U,permNum



def determinent(A):
    shape=A.shape
    if shape[0]!=shape[1]:
        raise Exception("Determinent calculation requires square matrix")
    P,L,U,permNum=lu(A)
    det=1

    for i in range(shape[0]):
        det=U[i,i]*det


    det=det*(-1)**(permNum)
    return det,U,L

print(A)
det=determinent(A)
print(det)

        
def cramer(A,b):
    b=b.reshape((b.size))
    shape=A.shape
    x=np.ones((shape[1],1))
    if shape[0]!=shape[1] or shape[0]!=b.size:
        print("Error: determinent requires square matrix")
        return x
    detA=determinent(A)
    for i in range(A.shape[1]):
        B=A.copy()
        B[:,i]=b
        detB=determinent(B)
        x[i,0]=detB/detA

    return x





def rref(A,Upper=False):
    if Upper==False:
        R,permNum=rr(A)
    rows=R.shape[0]
    columns=R.shape[1]
    maxPivots=min(rows,columns)
    for i in range(0,maxPivots):
        for j in range(i,columns):
            pivot=R[i,j]
            if abs(pivot)<10**(-10):
                continue
            R[i,j:]=R[i,j:]/pivot
            for k in range(i-1,-1,-1):     
                if abs(R[k,j])<10**(-10):
                    continue
                multiplier=-R[k,j]
                R[k,j:]=R[k,j:]+R[i,j:]*multiplier
            break
    return R


#A=np.array([[1,1,1,1],[5,2,3,4],[1,3,6,10],[1,4,10,20]])

def inv(A):
    rows=A.shape[0]
    columns=A.shape[1]
    if rows!=columns:
        raise Exception("Matrix must be square to be invertible")
    Ainv=np.concatenate((A,np.identity(rows)),axis=1)
    GaussJordian=rref(Ainv)
    I=GaussJordian[:,:rows]
    I=np.array(I,dtype=int)
    for i in range(rows):
        if I[i,i]!=1:
            raise Exception("Singular Matrix")
    return(GaussJordian[:,rows:])
    

def solveAxb(A,b):
    x=np.zeros((A.shape[1],1))
    Ab=np.concatenate((A,b),axis=1)
    R,permNum=rr(Ab)
    rows=R.shape[0]
    columns=R.shape[1]
    null=list(range(columns-1))

    for i in range(rows-1,-1,-1):
        for j in range(i,columns):
            pivot=R[i,j]
            if pivot==0:
                continue
            elif j==columns-1:
                raise Exception("No Solution")
            R[i,j:]=R[i,j:]/pivot
            null.remove(j)
            for k in range(i-1,-1,-1):     
                if R[k,j]==0:
                    continue
                multiplier=-R[k,j]
                R[k,j:]=R[k,j:]+R[i,j:]*multiplier
            x[j,0]=R[i,columns-1]
            break

    
    return x
                                                                    
    
##A=np.array([[1,1,1,1],[1,1,0,1],[0,0,0,1],[1,2,0,0]])
##A=np.random.rand(100,100)
##print(A)
##r=inv(A)
##print(r-np.linalg.inv(A))
                    
                

##S1=np.array([[-1/6**.5,3/34**0.5,-2**0.5/2,1],[-1/6**.5,3/34**0.5,2**0.5/2,0],[2/6**.5,4/34**.5,0,0]])
##c=rref(S1)[:,3].reshape((3,1))
##S=S1[:,:3]
##lamb=np.identity(3)
##lamb[0,0]=0
##lamb[2,2]=0.2


##u100=c[1,0]*S[:,1]
##print(u100)

##A=np.array([[0,1,1,1,1,5],[0,1,2,3,5,1],[0,0,1,1,0,1],[0,0,0,0,-5,-2]])
##b=np.array([[6],[2],[1],[-2]])
##print(solveAxb(A,b))





##start=time.perf_counter()
##count=0
##for i in range(3,17):
##    for k in range(100):
##        count+=1
##        A=np.random.randint(-5,6,size=(i,i))
##        b=np.random.randint(-5,6,size=(i,1))
##        try:
##            mine=solveAxb(A,b)
##            nums=np.linalg.solve(A,b)
##            diff=np.sum(mine-nums)
##            if diff>1:
##                print(A)
##                print(mine)
##                print(nums)
##
##                break
##        except:
##            try:
##                nums=np.linalg.inv(A)
##                print("mine failed")
##                print(A)
##                print(count)
##            except:
##                pass
##        
##
##
##
##print(time.perf_counter()-start)

##start=time.perf_counter()
##for i in range(100,101):
##    for j in range(100):
##        A=np.random.randint(-5,6,size=(i,i))
##        
##A=np.array([[-5, -4, -5, -3, -1, -4],
## [ 3,  1,  3,  1,  4, -1],
## [-2, -4, -3, -1, -5,  5],
## [ 3, -5,  4, -4,  3,  2],
## [ 3, -1, -5,  3, -4, -5],
## [ 4, -1,  1, -2,  2, -3]])

##print(lu(A))
##
##print(time.perf_counter()-start)
#shape r is always nxn and shape(Q)=shape(A)
