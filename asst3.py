import numpy as np
a = [1]*10
h =  0.75/(10+1)
r0 = 1
K = 2*np.pi*h

def objectiveFunction(x,h,r0):
    return K*x[0]*fi(x[0],r0,h) + K*sumOfFi(x,h) + K*x[-1]*fi(x[-1],r0,h)

# x is a vector
def sumOfFi(x,h):
        return sum (x[i]*fi(x[i],x[i+1],h) for i in range(len(x)-1))


#x1 x2 are values
def fi(x1,x2,h):
    return (1+((x2-x1)/h)**2)**0.5


def grad(x,h,r0):
    vectorGrad = [0.0]*len(x)
    c0 = (1+ ((r0-x[0])/h)**2)**0.5
    vectorGrad[0] = c0 - ((r0-x[0])*x[0])/((h**2)*c0)
    c1 = (1+ ((r0-x[-1])/h)**2)**0.5
    vectorGrad[-1] = c1 - ((r0-x[-1])*x[-1])/((h**2)*c1)
    print "lastElement:",vectorGrad[-1]

    for i in range (0,len(x)-1):
        tmp_i,tmp_i1 = d_fi(x[i],x[i+1])
        vectorGrad[i] += tmp_i
        vectorGrad[i+1] += tmp_i1

        print tmp_i , tmp_i1

    print "lastElement v2", vectorGrad[-1]
    return np.array(vectorGrad)*K

def d_fi(x0,x1):
    c = (1+ ((x0-x1)/h)**2)**0.5
    return ( -(x0*(-x0+x1))/((h**2)*c)+ c , x0*(x0-x1)/((h**2)*c) )

#print (objectiveFunction(a,h,r0))
print grad(a,h,r0)

def verify():
    b = [1] * len(a)
    b[-1] = a[-1] + 0.0001
    print (objectiveFunction(b,h,r0)-objectiveFunction(a,h,r0))/0.0001

ayre()
