import numpy as np
import scipy
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import sys

class GeneralTrace(object):
    def __init__(self,n_points=9000,num_transit_ticks=40,t_stop=3.):
        self.t = np.linspace(0.,t_stop,n_points)
        self.dt = self.t[1]-self.t[0]
        self.angle_bin_size_rad = 2*np.pi/num_transit_ticks
        self.angle_bin_size_deg = np.degrees(2*np.pi/num_transit_ticks)
        self.location_bins_rads = np.arange(-np.pi,np.pi,self.angle_bin_size_rad)
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
        reversals_inds = np.where(np.diff(np.signbit(np.diff(self.path))))

        reversal_thetas = self.path[reversals_inds]

        bins_rads = self.location_bins_rads
        reversal_bin_counts,bins = np.histogram(reversal_thetas,bins=bins_rads)
        reversal_bin_counts[np.isnan(reversal_bin_counts)] = 0.

        return reversals_inds,reversal_bin_counts

    def find_reversals_peaks(self,order=71,dis=30,wid=200):
        th_filter = scipy.signal.savgol_filter(np.unwrap(self.path),1*order,1)
        peaks_p, properties = find_peaks(th_filter, height=None, threshold=None, distance=dis*1, width = wid)
        peaks_n, properties = find_peaks(-th_filter, height=None, threshold=None, distance=dis*1, width = wid)
        peaks_inds = np.hstack((peaks_p,peaks_n))

        reversal_thetas = self.path[peaks_inds]

        bins_rads = self.location_bins_rads
        reversal_counts,bins = np.histogram(reversal_thetas,bins=bins_rads)
        reversal_counts[np.isnan(reversal_counts)] = 0.

        return peaks_inds,reversal_counts


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




# class EmpiricalTrace(object):
#     def __init__(self,t_start=0,t_stop=3,num_transit_ticks=24,):
#         self.t = np.linspace(t_start,t_stop,n_points)
#         self.dt = self.t[1]-self.t[0]
#         self.angle_bin_size_rad = 2*np.pi/num_transit_ticks
#



class ExpandingSinusoidTrace(GeneralTrace):

    def __init__(self,theta_0,a,omega,n_points=9000,
        num_transit_ticks=40):

        #setup all inherited properties
        super().__init__(n_points=n_points,num_transit_ticks=num_transit_ticks)

        #plus:
        envelope = a*np.exp(self.t)
        self.theta_0 = theta_0
        #Want this to still work when there are multiple rows (trials) of traces
        self.path = (np.pi/16)*np.sin((2*np.pi*omega)*self.t)*envelope+self.theta_0
        #Then store a version of this array where the bottom is 0 and angles are degrees
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)

class LinearExpandingSinusoidTrace(GeneralTrace):
    def __init__(self,theta_0,T1,T2,f1,f2,n_points=9000,
        num_transit_ticks=40,a11=0,a12=1):
        #t0: t value at which slope of expansion changes
        #m1: first slope of expansion
        #m2: second slope of expansion
        #omega: frequency
        #a11: nonzero start amplitude

        #setup all inherited properties
        super().__init__(n_points=n_points,num_transit_ticks=num_transit_ticks)

        self.theta_0 = theta_0
        #plus:

        m1 = (a12-a11)/T1
        m2 = (np.pi-(a12))/(T2)
        self.envelope = a11 + m1*self.t
        self.envelope[self.t>T1] = m2*(self.t[self.t>T1]-T1) +m1*T1+a11
        #Want this to still work when there are multiple rows (trials) of traces
        freq_over_time = np.zeros_like(self.envelope)
        freq_over_time[self.t<=T1] = f1
        freq_over_time[self.t>T1] = f2
        self.path = np.sin((np.pi*freq_over_time)*self.t)*self.envelope+self.theta_0
        #Then store a version of this array where the bottom is 0 and angles are degrees
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)


class TwoFreqSinusoidTrace(GeneralTrace):

    def __init__(self,theta_0,a,b,n,omega,n_points=9000,
        num_transit_ticks=40):
        #setup all inherited properties
        super().__init__(n_points=n_points,num_transit_ticks=num_transit_ticks)
        self.theta_0 = theta_0
        self.path = self.theta_0 + b*np.sin(n*(2*np.pi*omega)*self.t)+a*np.sin((2*np.pi*omega)*self.t)
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)

class SinPlusSquareWave(GeneralTrace):

    def __init__(self,theta_0,a,b,n,omega,n_points=9000,
        num_transit_ticks=40):

        #setup all inherited properties
        super().__init__(n_points=n_points,num_transit_ticks=num_transit_ticks)

        self.theta_0 = theta_0
        self.path = self.theta_0 + b*np.sin(n*(2*np.pi*omega)*self.t)+a*signal.square((2*np.pi*omega)*self.t)
        self.path_origin_shifted_degrees = (self.path - theta_0)*(180/np.pi)

class DistanceListTrace(GeneralTrace):
    #Like a GeneralTrace, except that is specifically given by a list of
    #straight line inter-reversal distances, with a constant velocity.

    def __init__(self,food1_loc,ird_values,r_0,velocity=3,n_points=9000,
        num_transit_ticks=40,t_stop=3.):

        #k_histogram is a

        #setup all inherited properties
        super().__init__(n_points=n_points,num_transit_ticks=num_transit_ticks,t_stop=t_stop)

        self.velocity = velocity    ## mm/s
        self.food1_loc = food1_loc
        self.ird_list = ird_values
        #print(ird_values)
        self.straight_path_times = ird_values/self.velocity
        #print(self.straight_path_times/ird_values)
        #print(velocity)
        #compute a cumsum to get reversal times
        #switch above to mins
        self.straight_path_times/=60.
        self.reversal_times = np.cumsum(self.straight_path_times)


        #also crop the ir distances this way
        self.ird_list = self.ird_list[0:len(self.straight_path_times)-1]
        self.ird_list = np.hstack((np.array(r_0),self.ird_list))
        alternating_signs = np.ones_like(self.ird_list)
        alternating_signs[0::2] = -1.
        self.delta_theta_coarse = alternating_signs*self.ird_list
        self.delta_theta_coarse = np.hstack((np.array([food1_loc]),self.delta_theta_coarse))
        self.theta_coarse = np.cumsum(self.delta_theta_coarse)

        t_interpolate = np.hstack((np.array([0]),self.reversal_times))

        # plt.figure()
        # plt.plot(t_interpolate,self.theta_coarse,'ro-')

        self.path = np.interp(self.t,
            t_interpolate, self.theta_coarse)

        # plt.plot(self.t,self.path,'b')

        # plt.xlim([0,max(self.t)])

        # plt.show()

        self.path_origin_shifted_degrees = (self.path)*(180/np.pi)


        # plt.show()


    #Draw list of inter-reversal distances from histogram

    #Linear interpolation back and forth using these distances and slope given
    #by the velocity (fixed)


class Model4Trace(DistanceListTrace):

    def __init__(self,food_case,s,r_0,k_std,eps_mean,eps_std,n_points=9000,
        num_transit_ticks=40,velocity=np.radians(10)):
        #r_0: excursion distance
        #k: multiplicative constant
        #g: draw from a distribution: (80,15,5)
        #epsilon: characteristic search length
        #d: food case-dependent distance


        #Use case to define d
        if food_case==1:
            d = 0.

        if food_case==2:
            d = np.pi/3

        if food_case==3:
            d = np.pi/4

        if food_case==4:
            d = 2*np.pi/3

        num_steps = 100

        #Desired output: list of reversal locations (theta) in order

        ks = np.random.normal(1.,k_std,size=num_steps)

        f2 = s+d

        self.d = d

        #(0)
        #Start at r_0 to the left of last food
        #Move r_0 to the left (should be updated to be either left or right, equal probability)***************
        current_loc = s + (d/2)+r_0/2
        current_loc = s + r_0/2

        reversal_locations = [current_loc]

        for step in range(num_steps):
    #        print('k:',ks[step])
#            print('step:',step)
#            print('theta:',np.degrees(current_loc))

            if (current_loc>s)&(current_loc<s+d/2):
#                print('a')
                current_loc-= ks[step]*r_0
            elif (current_loc<s)&(current_loc>s-d):
#                print('b')
                current_loc+= ks[step]*r_0
            elif (current_loc>s+(d/2))&(current_loc<s+d):
#                print('c')
                current_loc+= ks[step]*r_0
            elif (current_loc>s+d)&(current_loc<s+2*d):
#                print('d')
                current_loc-= ks[step]*r_0
            else:
                current_loc+=np.radians(np.random.uniform(-10,10))







            #(2)
            #Return to first food (set position to 0)
            #(3)
            #Draw k. Move right (or opposite of last step) k_1*r_0.



#            current_loc+=r_0+ks[step]*r_0

            #(4)
            #Draw g according to [80,15,5] [1,0,-1]. (can be time-varying)
            probs = np.array([.9,.08,.02])
            #probs = np.array([0.,1.,0.])
        #    probs = np.array([1.,0.,0.])
            g = np.random.choice(np.array([1,0,-1]),p=probs)

            epsilon = np.random.normal(eps_mean,eps_std)
        #    print(g)
    #        print(current_loc)
            if g==1:
                #g=1 then move d+episilon right (toward F2).
                #print(current_loc)
                #sys.exit()
                if (current_loc>0)&(current_loc<d):
                #    print('here1')
                    current_loc += d+epsilon
                elif current_loc>d:
                #    print('here2')
                    current_loc -= d+epsilon
    #        print(current_loc)
            #d = abs(F2) = abs(F1-F2)
            #epsilon ~ N(eps_mean,eps_std)

            #g=-1 move d+epsilon left
            if g==-1:
                if current_loc<=d:
                    current_loc -= d+epsilon
                if current_loc>d:
                    current_loc += d+epsilon

            #(5) Store r_1.

            reversal_locations.append(current_loc)

        self.reversal_thetas = np.array(reversal_locations)
        inter_reversal_distances = np.abs(np.diff(self.reversal_thetas))



        velocity = np.radians(10)

        print(reversal_locations[0])

        super().__init__(reversal_locations[0],inter_reversal_distances,
            reversal_locations[0],1,1,velocity=velocity,n_points=9000,num_transit_ticks=40)
        self.velocity = velocity
