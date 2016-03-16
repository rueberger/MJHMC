import numpy as np
from find_best_params import find
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb

if __name__=="__main__":
    print('MJHMC Path')
    print(sys.argv[1])
    print('Control Path')
    print(sys.argv[2])
    print('Save Path')
    print(sys.argv[3])
    #Calculating logs for mjhmc
    mjhmc = find(sys.argv[1],0)
    #Calculating logs for control
    control = find(sys.argv[2],0)
    #Plotting Epsilon
    plt.plot(mjhmc[:,1],mjhmc[:,0],'r+',label='MJHMC')
    plt.hold(True)
    plt.plot(control[:,1],control[:,0],'g*',label='Control')
    plt.xlabel('Epsilon')
    plt.ylabel('Obj. fn. value')
    plt.legend()
    plt.title('Obj fn vs epsilon') 
    plt.savefig(sys.argv[3]+'epsilon.png')
    print('Epsilon PNG Complete')
    #CLF
    plt.clf()
    #Plotting  Beta
    plt.plot(mjhmc[:,2],mjhmc[:,0],'r+',label='MJHMC')
    plt.hold(True)
    plt.plot(control[:,2],control[:,0],'g*',label='Control')
    plt.xlabel('Beta')
    plt.ylabel('Obj. fn. value')
    plt.title('Obj fn vs Beta') 
    plt.legend()
    plt.savefig(sys.argv[3]+'beta.png')
    print('Beta PNG Complete')
    #CLF
    plt.clf()
    #Plotting M 
    plt.plot(mjhmc[:,3],mjhmc[:,0],'r+',label='MJHMC')
    plt.hold(True)
    plt.plot(control[:,3],control[:,0],'g*',label='Control')
    plt.xlabel('Num Leap Frog Steps')
    plt.ylabel('Obj. fn. value')
    plt.title('Obj fn vs M') 
    plt.legend()
    plt.savefig(sys.argv[3]+'M.png')
    print('Num Steps PNG Complete')
