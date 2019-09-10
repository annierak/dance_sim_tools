import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

def histogram_draw_to_parameters(h,tao,alpha,beta,bin_width=15):
    #h is a vector of length = number of bins
    #Return a11,a12,T1,T2,f1,f2, the parameters for generation of a fake trace

    T1 = alpha*tao
    T2 = beta*tao

    bins = np.arange(0,2*np.pi+np.radians(bin_width),np.radians(bin_width))

    # order=71
    # dis=30
    # wid=200
    # peak_inds,_ = find_peaks(h, height=None, threshold=None, distance=dis*1, width = wid)

    h_sorted = np.unique(np.sort(h))[::-1]
    #Currently this is HARD CODED for 2 symmetric peaks of the SAME VALUE. Need to generalize.
    #Actually I think it should work for two highest values, not just symmetric peaks.
    peak_ind_1 = np.where(h==h_sorted[0])[0][0]
    try:
        peak_ind_2 = np.where(h==h_sorted[0])[0][1] #first the case where two identical maxima
    except(IndexError):
        peak_ind_2 = np.where(h==h_sorted[1])[0][0] #otherwise the first time the second highest value occurs
    peak_inds = np.array([peak_ind_1,peak_ind_2])

    B_1a = bins[peak_ind_1]-bin_width/2
    B_1b = bins[peak_ind_2]-bin_width/2
    # print(np.random.uniform(B_1a,B_1b,size=2))
    a11,a12 = np.random.uniform(B_1a,B_1b,size=2)

    B_1_reversal_total = np.sum(h[peak_inds]) #this is both sides

    # print('B_1_reversal_total',B_1_reversal_total)

    f1 = B_1_reversal_total/T1

    B_2_reversal_total = np.sum(h[(peak_inds[-1]+1):])

    f2 = B_2_reversal_total/T2


    return a11,a12,T1,T2,f1,f2


#Testing the above function
# h = np.random.choice(np.arange(1,10,1),size=(12))
# h[3] = 20
# h = np.hstack((h[::-1],h))
#
# tao = 3
# alpha = 0.7
# beta = 0.4
# bin_width=15
#
# bins = np.arange(0,2*np.pi+np.radians(bin_width),np.radians(bin_width))
#
# plt.figure()
# plt.bar(bins[:-1],h,width=np.radians(bin_width))
# a11,a12,T1,T2,f1,f2 = histogram_draw_to_parameters(h,tao,alpha,beta,bin_width=bin_width)
# plt.show()

def draw_reversal_distances_tm(num_traces,thresh,velocity,dt,kappa):

    #This is the threshold crossing mechanism explained in neural_model_0.ipynb.
    #returns a sample of num_traces distances at which the threshold was crossed
    E_i_1 = np.zeros(num_traces)

    distances = np.full(num_traces,np.nan)
    cum_deltas = np.zeros(num_traces) #This will be the same for every trace until the true
    already_crossed_inds = np.zeros(num_traces).astype(bool)
    #motion is set to vary (true delta_i)

    i = 0
    #Loop through time steps starting from first reversal
    while np.sum(already_crossed_inds==0)>0:
        #Each loop:
        #set a true #/Delta/theta$ or $/Delta d_{LR}$ (signal acquired about body motion
        #of previous step)
        #(This can be the same each iteration, but can vary across traces and time steps)
        delta_i = velocity*dt
        cum_deltas+=delta_i
        #Add, to E_{i-1}
        #the true /Delta/theta or /Delta d_{LR},
        E_i  = E_i_1 +delta_i
        #Add to the above a draw from a r.v. with mean 0 and variance = kE_{i-1} --> this is E_i.
        E_i = E_i + np.random.normal(0,kappa*E_i_1,num_traces)
        #(each iteration) check to see if E_i has crossed (0 or r_0 or L_F or |\theta-L_F|)
        #look at the set of E_is. the ones that have crossed the threshold set to nans so nothing
        #happens to them in the later iterations.
        inds_just_crossed = (E_i>thresh) & np.logical_not(already_crossed_inds)
        E_i[inds_just_crossed] = np.nan
        #and the location it's at when the threshold is crossed
        distances[inds_just_crossed] = cum_deltas[inds_just_crossed]
    #    #reassign E_i_1 to what E_i currently is
        E_i_1 = E_i
        already_crossed_inds = already_crossed_inds | inds_just_crossed
        i+=1
    
    return distances
