import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from Visualisation import read_image
import scipy.interpolate as it

def fft_shape_analysis(n1,n2,n_fft):
    
    X_all=[]
    for i in range(n1,n2):
        for j in range(3):
            # print(i,j)
            X=read_image(i,j)
            fft=fft_shape(X,n_fft)
            X_all.append(fft)
    return np.array(X_all)

def fft_shape(img,n_fft):
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imbw = cv.threshold(img,30,255,cv.THRESH_BINARY)
    
    # Run findContours - Note the RETR_EXTERNAL flag
    # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(imbw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(contours) > 1:
        s = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > s:
                s = contours[i].shape[0]
                max_ele = i
        contours = [contours[max_ele]]            
    
    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(img)
    
    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv.drawContours(out, contours, -1, 255, 3)
    
    # # Spawn new windows that shows us the donut
    # # (in grayscale) and the detected contour
    # cv.imshow('Donut', img) 
    # cv.imshow('Output Contour', out)
    
    # trouver le centroid
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    # fonction d'interpolation
    cntr = (cnt.reshape(cnt.shape[0],cnt.shape[2])).T
    f = it.interp1d(np.arange(0,len(cnt)), cntr)
    
    # Changer le bords en 64 points
    cnt64 = f(np.linspace(0,len(cnt),n_fft+1)[:-1])
    
    # centroid distance
    cenDist = ((cnt64[0,:] - cx)**2 + (cnt64[1,:] - cy)**2)**0.5
    cenDistn = (cenDist - min(cenDist)) / (max(cenDist) - min(cenDist))
    
    # transformation discrete de Fourier
    fft = np.fft.fft(cenDistn)
    
    return np.abs(fft[1:int(fft.shape[0]/2)])

def contour_points_finder(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imbw = cv.threshold(img,60,255,cv.THRESH_BINARY)
    
    # Run findContours - Note the RETR_EXTERNAL flag
    # Also, we want to find the best contour possible with CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(imbw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    max=0
    for i in range(len(contours)):
        if contours[i].shape[0]>max:
            i_max=i
            max=contours[i].shape[0]
    points=np.reshape(contours[i_max],(contours[i_max].shape[0],contours[i_max].shape[2]))
    return points

def angular_relative_position(pts):
    pt0=pts[0]
    center=np.array([np.mean(pts[:,0]),np.mean(pts[:,1])])
    x1=pt0-center
    theta=[]
    for i in range(pts.shape[0]):
        # a=np.linalg.norm(pts[i]-center)
        # b=np.linalg.norm(pt0-center)
        # c=np.linalg.norm(pts[i]-pt0)
        # theta.append(np.arccos((a**2+b**2-c**2)/(2*a*b)))
        x2=pts[i]-center
        cos_theta=np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
        det=x1[0]*x2[1]-x1[1]*x2[0]
        if det>0:
            theta.append(-np.arccos(cos_theta)+2*np.pi)
        else:
            theta.append(np.arccos(cos_theta))
            
    j=0
    for i in range(0,pts.shape[0]-1):
        theta[i]=theta[i]+2*j*np.pi
        if theta[i+1]-theta[i]>2*np.pi:
            j-=1
        elif theta[i+1]-theta[i]<-2*np.pi:
            j+=1
    
    return np.array(theta)
    
def reorder_points(pts,theta):
    index=np.argsort(theta)
    theta_sorted=np.sort(theta)
    pts_sorted=np.empty((0,2))
    for i in range(pts.shape[0]):
        pts_sorted=np.vstack((pts_sorted,pts[index[i]]))
    return pts_sorted,theta_sorted
    
def salience_relevance(pts,K):
    
    if (np.float(K)<pts.shape[0]/2) and (K>0):
        sal_relev=np.zeros(pts.shape[0]-2*K)
        for i in range(K,pts.shape[0]-K):
            a=pts[i+K,0]-pts[i,0]
            b=pts[i+K,1]-pts[i,1]
            c=pts[i+K,0]-2*pts[i,0]+pts[i-K,0]
            d=pts[i+K,1]-2*pts[i,1]+pts[i-K,1]
            sal_relev[i-K]=(a*d-b*c)/((a**2+b**2)**(3/2))
        return np.array(sal_relev)
            
    else:
        print("Warning: choose another value for K")
        return(np.array([]))

def reduce_pts(pts1,pts2):
    l_min=min(len(pts1),len(pts2))
    if l_min==len(pts1):
        pts_min=pts1
        pts_max=pts2
    else:
        pts_min=pts2
        pts_max=pts1
    l_max=len(pts_max)
    
    pts1=np.zeros((l_min,2))
    pts2=np.zeros((l_min,2))
    
    c=0
    for i in range(l_max):
        if i/l_max >= c/l_min:
            pts1[c]=pts_min[c]
            pts2[c]=pts_max[i]
            c+=1
    print(c)
    print(l_max)
    print(l_min)
    print(len(pts1))
    print(len(pts2))
    
    return pts1,pts2
    
            
    

# Test3:
"""
img1=read_image(251,0)
pts1=contour_points_finder(img1)

img2=read_image(251,1)
pts2=contour_points_finder(img2)

pts1,pts2=reduce_pts(pts1,pts2)

theta1=angular_relative_position(pts1)

plt.figure()
plt.plot(theta1)
plt.title("Angular Relative Position 1")

theta2=angular_relative_position(pts2)

plt.figure()
plt.plot(theta2)
plt.title("Angular Relative Position 2")

plt.show()
"""
"""
# Test2:

img=read_image(251,0)
pts=contour_points_finder(img)
theta=angular_relative_position(pts)
sal=salience_relevance(pts,1)

plt.figure()
plt.imshow(img)

plt.figure()
plt.scatter(pts[:,0],pts[:,1])

plt.figure()
plt.plot(theta)
plt.title("Angular Relative Position")

plt.figure()
plt.plot(sal)
plt.title("Salience Relevance")

plt.show()


"""

"""
# Test1:
pts=np.array([[0,0],[1,0],[1,1],[0,1]])

pts=np.array([[0,0],[0.25,2],[1,0],[0.5,3],[2,0.7],[1,0.5],[1,1],[0,1]])
center=np.array([np.mean(pts[:,0]),np.mean(pts[:,1])])
plt.scatter(pts[:,0],pts[:,1])
plt.plot(center[0],center[1],'ro')
plt.show()
K=1
theta=angular_relative_position(pts)
print(theta)
pts, theta= reorder_points(pts,theta)
print(pts)
print(theta)
sal_rev=salience_relevance(pts,K)
print(sal_rev)

"""