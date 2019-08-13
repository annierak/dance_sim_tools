import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt

class GeneralTrace(object):
    def __init__(self,n_points=9000,t_start=0,t_stop=3,num_transit_ticks=40):
        self.t = np.linspace(t_start,t_stop,n_points)
        self.dt = self.t[1]-self.t[0]
        self.angle_bin_size_rad = 2*np.pi/num_transit_ticks
        self.angle_bin_size_deg = np.degrees(2*np.pi/num_transit_ticks)
        self.location_bins_rads = np.arange(0,2*np.pi,self.angle_bin_size_rad)
        self.location_bins_degs = np.arange(-180,180,np.degrees(self.angle_bin_size_deg))
        #currently this doesn't include 2pi as the n+1 bin edge
    def find_reversal_inds(self):

        #old version:  find where the derivative is small enough (not very effective)
        # reversal_bool = (np.abs(np.diff(self.path))<1e-3)
        # reversal_inds = np.where(reversal_bool)[0]
        # non_duplicate_inds = np.diff(np.hstack((np.array(0),reversal_inds)))>1
        # reversal_inds = reversal_inds[non_duplicate_inds]
        #Want this to still work when there are multiple rows (trials) of traces

        #second version: find every time the derivative changes sign

        deriv_zero_crossings = np.where(np.diff(np.signbit(np.diff(self.path))))
        return deriv_zero_crossings

    def reversal_loc_hist(self,shifted=False):

        reversal_inds = self.find_reversal_inds()
        if shifted:
            bins =self.location_bins_degs
            n,bins = np.histogram(self.path_origin_shifted_degrees[reversal_inds],bins=bins)
        else:
            bins = self.location_bins_rads
            n,bins = np.histogram(self.path[reversal_inds],bins=bins)
        return n,bins

    def crosses_0(self,lower_bound,upper_bound):
        if type(lower_bound)==float:
            lower_bound = np.array([[lower_bound]])
            upper_bound = np.array([[upper_bound]])

        #Returns
        #(0) indices that the (1D) path crossed the upper bound
        #(1) indices that the (1D) path crossed the lower bound
        #(2) cross_sign, a vec that is only nonzero at upper or lower bound
        #crosses, and gives the sign of the derivative at that point

        deriv = np.gradient(self.path)
        epsilon = np.abs(deriv)/2
        upper_bound_crosses = (np.abs(self.path[None,:]-upper_bound[:,None])<epsilon[None,:]).astype(bool)
        lower_bound_crosses = (np.abs(self.path[None,:]-lower_bound[:,None])<epsilon[None,:]).astype(bool)

        cross_sign =(upper_bound_crosses
            | lower_bound_crosses).astype(int)*np.sign(deriv)

        return upper_bound_crosses,lower_bound_crosses,cross_sign

        #shape (num_transit_ticks [=n_bins] x n_points)

    def official_crosses(self,shifted=False,demo_bin=None):
        #Returns, for each location bin, the number of times the trace officially
        #crossed that bin, by the criteria that:

        #A portion of a trace counts as a 'cross' if it
        #either {passes through upper_bound decreasing  and
        #immediately passes through lower_bound decreasing}
        #or {passes through lower_bound increasing  and
        #immediately passes through upper_bound increasing}

        #Basically we are going to need to call self.crosses_0() for each individual bin loc

        #(0): list out the bins (by their bottom edge) according to self.num_transit_ticks
        #(1) set up the array of each bins' relevant upper and lower bounds (for crossing)
        #shape: (num_transit_ticks x 2)

        if shifted:
            bins_degs =self.location_bins_degs
            bin_bounds =  np.vstack((bins_degs-self.angle_bin_size_deg,bins_degs+2*self.angle_bin_size_deg)).T
        else:
            bins_rads = self.location_bins_rads
            bin_bounds =  np.vstack((bins_rads-self.angle_bin_size_rad,bins_rads+2*self.angle_bin_size_rad)).T

        #(2) for each row of ^, send the bounds through self.crosses_0() to find the
        #upper_bound_crosses, lower_bound_crosses, cross_sign
        #output will then be 3 arrays
        #of shape (num_transit_ticks x n_points)
        #--call them upper_bound_crosses_1,lower_bound_crosses_1,cross_sign_1 --

        #bin_bounds is of shape (n_bins,2)

        #the vector we want to pass to crosses_0 should be (n_bins,2)
        upper_bound_crosses_1,lower_bound_crosses_1,cross_sign_1 = \
            self.crosses_0(bin_bounds[:,0],bin_bounds[:,1])

        #then for each transit_tick row, move through the time direction (columns)
        #and flag each *official* cross (add it to a cumulating sum)
        #

        all_zero_cols = (np.abs(cross_sign_1).astype(bool)) #shape is (n_bins,n_points)
        #this variable is just a bool of whether there's any type of cross

        m,n = np.where(all_zero_cols)
        n_max_crosses = np.max(np.sum(all_zero_cols,axis=1)) #maximum cross count for a single bin
        all_zero_cols_inds = np.full((np.shape(upper_bound_crosses_1)[0],n_max_crosses),np.nan)
        #shape : (n_bins,n_max_crosses)


        for i in range(np.shape(all_zero_cols)[0]):
            to_add = n[(m==i)]
            all_zero_cols_inds[i,:len(to_add)] = to_add

        upper_bound_crosses_inds = np.full_like(all_zero_cols_inds,np.nan)
        lower_bound_crosses_inds = np.full_like(all_zero_cols_inds,np.nan)
        cross_sign_inds = np.full_like(all_zero_cols_inds,np.nan)

        for i in range(np.shape(all_zero_cols)[0]):
            to_add = upper_bound_crosses_1[i,all_zero_cols_inds[i,~np.isnan(all_zero_cols_inds[i,:])].astype(int)]
            upper_bound_crosses_inds[i,:len(to_add)] = to_add
            to_add = lower_bound_crosses_1[i,all_zero_cols_inds[i,~np.isnan(all_zero_cols_inds[i,:])].astype(int)]
            lower_bound_crosses_inds[i,:len(to_add)] = to_add
            to_add = cross_sign_1[i,all_zero_cols_inds[i,~np.isnan(all_zero_cols_inds[i,:])].astype(int)]
            cross_sign_inds[i,:len(to_add)] = to_add


        # print(upper_bound_crosses_inds[demo_bin,:])
        # print(lower_bound_crosses_inds[demo_bin,:])
        # print(cross_sign_inds[demo_bin,:])
        # #^ looks like it's working well

        #First, make a derivate of cross_sign_inds that tells us whether each
        #new element is the same as the previous one or different.
        cross_sign_1_flips = np.logical_not(np.diff(cross_sign_inds)==0.)
        cross_sign_1_flips = np.hstack((np.full((np.shape(all_zero_cols)[0],1),False),cross_sign_1_flips))
        # print(cross_sign_1_flips[demo_bin,:].astype(int))


        upward_crossings = (
            lower_bound_crosses_inds[:,:]==0)&(
                cross_sign_inds[:,:]==1.)&(
                np.logical_not(cross_sign_1_flips[:,:])) #this is a boolean where the number of entries is the number of crosses of any type

        downward_crossings = (
            lower_bound_crosses_inds[:,:-1]==0)&(
                cross_sign_inds[:,:-1]==-1.)&(
                np.logical_not(cross_sign_1_flips[:,1:]))
        #
        # print('------')
        # print('upward_crossings: '+str(upward_crossings[demo_bin,:].astype(int)))
        # print('downward_crossings: '+str(downward_crossings[demo_bin,:].astype(int)))


        #then, sum across the "crosses" dimension to get a list of crosses per bin
        upward_crossings,downward_crossings = np.sum(upward_crossings,axis=1),np.sum(downward_crossings,axis=1)


        return upward_crossings,downward_crossings


    def compute_transit_counts(self):
        #input shape: (trials x timestamps)
        #Return the binned transit counts
        #output shape: (trials x angle bins)
        return n

    def draw_trials(num_trials):
        #For the already inputted trace parameter values, draw a trace
        #num_trials times (drawing noise anew each trial),
        #and compute the transit vector for each trial
        #to return a 2d array of draws x theta (used for heatmap)

        return transits



class ExpandingSinusoidTrace(GeneralTrace):

    def __init__(self,theta_0,a,omega,n_points=9000,
        t_start=0,t_stop=3,num_transit_ticks=40):

        #setup all inherited properties
        super().__init__(n_points=n_points,t_start=t_start,t_stop=t_stop,num_transit_ticks=num_transit_ticks)

        #plus:
        envelope = a*np.exp(self.t)
        self.theta_0 = theta_0
        #Want this to still work when there are multiple rows (trials) of traces
        self.path = (np.pi/16)*np.sin((2*np.pi*omega)*self.t)*envelope+self.theta_0
        #Then store a version of this array where the bottom is 0 and angles are degrees
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)

class LinearExpandingSinusoidTrace(GeneralTrace):
    def __init__(self,theta_0,a,omega,t0,m1,m2,n_points=9000,
        t_start=0,t_stop=3,num_transit_ticks=40,y0=0):
        #a: scale
        #t0: t value at which slope of expansion changes
        #m1: first slope of expansion
        #m2: second slope of expansion
        #omega: frequency
        #y0: nonzero start amplitude


        #setup all inherited properties
        super().__init__(n_points=n_points,t_start=t_start,t_stop=t_stop,num_transit_ticks=num_transit_ticks)

        #plus:
        envelope = y0 + m1*self.t
        envelope[self.t>t0] = m2*(self.t[self.t>t0]-t0) +m1*t0+y0
        envelope = a*envelope
        #Want this to still work when there are multiple rows (trials) of traces
        self.path = (np.pi/16)*np.sin((2*np.pi*omega)*self.t)*envelope+self.theta_0
        #Then store a version of this array where the bottom is 0 and angles are degrees
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)


class TwoFreqSinusoidTrace(GeneralTrace):

    def __init__(self,theta_0,a,b,n,omega,n_points=9000,
        t_start=0,t_stop=3,num_transit_ticks=40):
        #setup all inherited properties
        super().__init__(n_points=n_points,t_start=t_start,t_stop=t_stop,num_transit_ticks=num_transit_ticks)
        self.theta_0 = theta_0
        self.path = self.theta_0 + b*np.sin(n*(2*np.pi*omega)*self.t)+a*np.sin((2*np.pi*omega)*self.t)
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)
class SinPlusSquareWave(GeneralTrace):

    def __init__(self,theta_0,a,b,n,omega,n_points=9000,
        t_start=0,t_stop=3,num_transit_ticks=40):

        #setup all inherited properties
        super().__init__(n_points=n_points,t_start=t_start,t_stop=t_stop,num_transit_ticks=num_transit_ticks)

        self.theta_0 = theta_0
        self.path = self.theta_0 + b*np.sin(n*(2*np.pi*omega)*self.t)+a*signal.square((2*np.pi*omega)*self.t)
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)
