import numpy as np
import random as rnd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import imp, heapq, time, sys#, os
#from scipy.io import netcdf
#import matplotlib.colors as mcolors
#import matplotlib as mpl

imp.load_source('helper','helper.py')
from helper import *

class LIF_neuron:
    
    def __init__(self,state,postSyn,preSyn,paras):
        
        self.stateReset = paras['V_R']
        
        self.state = state       ## set initial voltage and current
        self.postSyn = postSyn
        self.preSyn = preSyn
        
        self.gamma = paras['gamma']            ## should be 1/3, 1/2, 2 or 3
        self.Iext = paras['Iext']
        self.STDP_tau = paras['STDP_tau_pos']*3
        
        self.ct_spikes = 0
        self.lastSpikeTime = np.NaN
        
        self.calcSpikeTime = True
        
        
        
    def evolve_dt(self,dt,T):
        
        expM = np.exp(-dt)
        expS = expM**self.gamma						# power function depending on gamma
        
        tmpI = self.state[1]/(1 - self.gamma)
        
    	# evolve the state between spikes
        self.state[0] =  1 + self.Iext + tmpI*expS - (1 + self.Iext + tmpI - self.state[0])*expM		# Eq. (4.8)
        self.state[1] *= expS
        
        if (T-self.lastSpikeTime)>self.STDP_tau:
            self.lastSpikeTime = np.NaN
#        rm_time = np.where(np.array(self.lastSpikeTime)>self.STDP_tau)[0]
#        for t in rm_time:
#            self.lastSpikeTime.pop(t)
            
        self.calcSpikeTime = True
        
        
    def evolve_spike(self,J):
        self.state[1] += J*self.gamma
        self.calcSpikeTime = True
        ## should save last spike time (or all within a certain window) to allow for STDP (symmetric?)
        
        
    def reset(self,T):
        self.state[0] = self.stateReset
        self.lastSpikeTime = T
        self.ct_spikes += 1
        
        self.calcSpikeTime = True
        
        
        
        
    def get_spikeTime(self):
        
        if self.calcSpikeTime:
            if (self.Iext <= 0) and (self.state[1] <= 0):			# this one is never gonna spike unless there is an excitatory input
                self.spikeTime = np.inf;
            else:
                # this is the numerical method to calculate the roots of a square function from the numerical recipes book
                # see thesis, p.~87
                a = -self.state[1]
                b = -(1 + self.Iext - self.state[0] - self.state[1])
                c = self.Iext
                d = -(b + np.sign(b)*np.sqrt(b*b - 4*a*c))/2
        
        		# This defines two solutions d/a and c/d. Only one of them can be physically correct.
                x1 = c/d
                x2 = d/a
        
        		# Check if the logarithm of this root makes sense and if the solution is unique.
                oneSol = False;
        
                if (x1 > 0) and (x1 <= 1):
                    oneSol = True
                    self.spikeTime = -np.log(x1)
                if (x2 > 0) and (x2 <= 1):
                    if oneSol:
        #				cout << "couldn't find a unique solution for this neuron with voltage=" << state[0] << " and current=" << state[1] << endl;
                        print("couldn't find a unique solution for this neuron")
                        sys.exit()
                    else:
                        oneSol = True
                        self.spikeTime = -np.log(x2)
        
        		# If both solutions are out of range, then the neuron will never spike until it gets another excitatory input, which
        		# is of course possible in the future evolution. For now, we'll set the spike time to infinity, though.
                if not oneSol:
                    self.spikeTime = np.inf
        
        self.calcSpikeTime = False
        
        return self.spikeTime
            

class network:
  
    def __init__(self,paras=[],options=False):
    
        self.set_params(paras)
        
        # construct network
        if options==False:
          options = {}
          options['A_mode'] = 'uniform'
          options['J_mode'] = 'gauss'
          
        self.create_topology(options)
        self.options = options
        
        if options['test']:
          self.test_topology()
        
        self.setup_neurons()
        self.ct_STDP = 0
        
    def simulate(self):
        
        self.Data = {}
        self.Data['spikeTrain'] = [[],[]]
        t_step = 1
        n_bins = 101
        self.Data['J_hist'] = np.zeros((int(self.paras['TC']/t_step+1),n_bins-1))
        self.Data['rate_hist'] = np.zeros((self.paras['N'],1))
        
        self.T = 0
        time_0 = time.time()
        t_ct = 0
        
        ### warmup
#        while self.T < self.paras['TW']:
#            self.iterate(False)      # could provide boolean to save stuff?
#        print('warmup done')
        plt.ion()
        
        f = plt.figure(figsize=(16,10))
        ax_train = f.add_subplot(3,1,1)
        ax_rate = f.add_subplot(6,1,3)
        ax_rate_hist = f.add_subplot(2,2,3)
        ax_J_hist = f.add_subplot(4,2,6)
        ax_J_diff_hist = f.add_subplot(4,2,8)
        
        ax_train.set_xlim([0,self.paras['TC']])
        ax_train.set_ylim([0,self.paras['N']])
        
        ax_rate.set_xlim([0,self.paras['TC']])
        ax_rate.set_ylim([0,1])
        ax_rate.set_xlabel('time [s]')
        ax_rate.set_ylabel('firing rate [Hz]')
        
        ax_J_hist.hist(self.J[self.A],np.linspace(-self.paras['STDP_J_max'],self.paras['STDP_J_max'],n_bins))
        ax_rate_hist.hist(self.Data['rate_hist'],n_bins)
        
        
        old_rate = np.zeros(self.paras['N'])
        plt.show(block=False)
        
        while self.T < self.paras['TC']:
            self.iterate(True)      # could provide boolean to save stuff?
            
            if self.T >= (t_ct*t_step):
                time_now = time.time()
                print('Time now: T = %3.1f, time elapsed: %3.1fs'%(self.T,time_now-time_0))
                print('maximum synaptic weight: J=%6.4f, mean synaptic weight: J=%6.4f'%(np.max(self.J),np.mean(self.J)))
                self.Data['J_hist'][t_ct,:] = np.histogram(self.J[self.A],np.linspace(-self.paras['STDP_J_max'],self.paras['STDP_J_max'],n_bins))[0]
                
                if self.options['test']==True:
                    ax_train.clear()
                    
                    ax_train.scatter(self.Data['spikeTrain'][1],self.Data['spikeTrain'][0],marker='|')
                    ax_train.set_xlim([0,self.paras['TC']])
                    ax_train.set_ylim([0,self.paras['N']])
                    
                    ax_J_hist.clear()
                    ax_J_hist.hist(self.J[:,:self.paras['N_E']][self.A[:,:self.paras['N_E']]],np.linspace(-self.paras['STDP_J_max'],self.paras['STDP_J_max'],n_bins),color='k')
                    ax_J_hist.hist(self.J[:,self.paras['N_E']:][self.A[:,self.paras['N_E']:]],np.linspace(-self.paras['STDP_J_max'],self.paras['STDP_J_max'],n_bins),color='r')
                    
                    ax_J_diff_hist.clear()
                    ax_J_diff_hist.plot(np.linspace(-self.paras['STDP_J_max'],self.paras['STDP_J_max'],100),self.Data['J_hist'][t_ct]-self.Data['J_hist'][0])
                    
                    for n in range(self.paras['N']):
                        self.Data['rate_hist'][n] = self.neuron[n].ct_spikes
                    
                    
                    ax_rate.plot(self.T,np.sum(self.Data['rate_hist'][:self.paras['N_E']]-old_rate[:self.paras['N_E']])/t_step/self.paras['N_E'],marker='.',color='k')
                    ax_rate.plot(self.T,np.sum(self.Data['rate_hist'][self.paras['N_E']:]-old_rate[self.paras['N_E']:])/t_step/self.paras['N_I'],marker='.',color='r')
                    
                    
                    old_rate = np.copy(self.Data['rate_hist'])
                    self.Data['rate_hist'] /= self.T
                    
                    ax_rate_hist.clear()
                    ax_rate_hist.hist(self.Data['rate_hist'][:self.paras['N_E']],np.linspace(0.,5.,n_bins),color='k')
                    ax_rate_hist.hist(self.Data['rate_hist'][:self.paras['N_I']],np.linspace(0.,5.,n_bins),color='r')
                    ax_rate_hist.set_xlim([0,2])
                    ax_rate_hist.set_xlabel('firing rate [Hz]')
                    
                    ax_J_diff_hist.set_xlabel('synaptic weight J')
                    ax_J_diff_hist.set_ylabel('difference')
                    
#                    print(len(self.Data['spikeTrain']))
#                    print(len(self.Data['spikeTrain'][0]))
#                    arr = p_train.get_offsets()
#                    print(arr)
#                    print(len(arr))
#                    arr=np.append(arr,[self.Data['spikeTrain'][1][-1],self.Data['spikeTrain'][0][-1]])
#                    p_train.set_offsets(arr)
                    plt.pause(0.1)
                t_ct += 1
                
                
            
        
#        t_steps = 1001
#        self.Data = {}
#        self.Data['measureTimes'] = np.linspace(0,self.paras['TC'],t_steps)
#        self.Data['phases'] = np.zeros((t_steps,self.paras['N']))
        
        
    
    def setup_neurons(self):        #include possibility to predetermine states
        
        states = np.random.uniform(self.paras['V_R'],self.paras['V_T'],(self.paras['N'],2)) # get random initial conditions
        self.neuron = [LIF_neuron(states[n],np.where(self.A[n,:])[0],np.where(self.A[:,n])[0],self.paras) for n in range(self.paras['N'])]
        
        
    def iterate(self,record):
        
        dt, n_spike = self.find_next_spike()    ## get time to evolve by finding lowest time to next spike
        self.evolve_dt(dt)                      ## evolve all neurons
        self.T += dt
        
        self.evolve_spike(n_spike)              ## evolve spike by updating postsynaptic and spiking neuron
        
        self.STDP(n_spike)
        
        if record:
            self.Data['spikeTrain'][0].append(n_spike)
            self.Data['spikeTrain'][1].append(self.T)
        
        
    def find_next_spike(self):
        
        spikeTime = np.inf
        n_spike = np.NaN
        
        for n in range(self.paras['N']):
            spikeTime_tmp = self.neuron[n].get_spikeTime()
            if spikeTime_tmp < spikeTime:
                spikeTime = spikeTime_tmp
                n_spike = n
            if spikeTime == 0:  ## if synchronous spiking, dt=0 can be updated immediately
                break
        
        return spikeTime, n_spike
    
    
    def evolve_dt(self,dt):
        
        for n in range(self.paras['N']):
            self.neuron[n].evolve_dt(dt,self.T)
    
    
    def evolve_spike(self,n):
#        postSyn = np.where(self.A[n])[0]               ## get indexes of neurons to be evolved from topology
        ## update all postsynaptic neurons
        for n_post in self.neuron[n].postSyn:
            self.neuron[n_post].evolve_spike(self.J[n_post,n])
            
        self.neuron[n].reset(self.T)                      ## reset spiking neuron
        
        
        
    def STDP(self,n):
        
        ## enable STDP only for excitatory? or reverse roles for inhibition?
#        if self.J[n1,n2]>0 and len(self.neuron[n1].lastSpikeTime) and len(self.neuron[n2].lastSpikeTime):
        for n_post in self.neuron[n].postSyn:
            dt = self.neuron[n_post].lastSpikeTime-self.neuron[n].lastSpikeTime
            if ~np.isnan(dt):
                
#                print("Time: %4.2f, times: %5.3f"%(self.T,dt))
#                print("weight pre : -> %6.4f   ,   <- %6.4f"%(self.J[n_post,n],self.J[n,n_post]))
                if self.J[n_post,n]:
                    self.ct_STDP += 1
                    self.J[n_post,n] += self.W_STDP(dt,self.J[n_post,n])
                if self.J[n,n_post]:
                    self.ct_STDP += 1
                    self.J[n,n_post] += self.W_STDP(-dt,self.J[n,n_post])
#                print("weight post: -> %6.4f   ,   <- %6.4f"%(self.J[n_post,n],self.J[n,n_post]))
                
        for n_pre in self.neuron[n].preSyn:
            dt = self.neuron[n].lastSpikeTime-self.neuron[n_pre].lastSpikeTime
            if ~np.isnan(dt):
                
#                print("Time: %4.2f, times: %5.3f"%(self.T,dt))
#                print("weight pre : -> %6.4f   ,   <- %6.4f"%(self.J[n,n_pre],self.J[n_pre,n]))
                if self.J[n,n_pre]:
                    self.ct_STDP += 1
                    self.J[n,n_pre] += self.W_STDP(dt,self.J[n,n_pre])
                if self.J[n_pre,n]:
                    self.ct_STDP += 1
                    self.J[n_pre,n] += self.W_STDP(-dt,self.J[n_pre,n])
#                print("weight post: -> %6.4f   ,   <- %6.4f"%(self.J[n,n_pre],self.J[n_pre,n]))
        
            
#            if top_fact == 1:
#                print('spike,post')
#            else:
#                print('spike,pre')
#            
            
#            
            
            ## multiplicative rescaling of all synaptic weights of this neuron
            ## but first: implement monitoring of synaptic weight distributions!
    
    def W_STDP(self,dt,J):
        
        if dt < 0:
            A = J * self.paras['STDP_eta_neg']
            return -A * np.exp(dt/self.paras['STDP_tau_neg'])
        else:
            A = np.sign(J)*(self.paras['STDP_J_max'] - np.abs(J)) * self.paras['STDP_eta_pos']
            return A * np.exp(-dt/self.paras['STDP_tau_pos'])
        
    
    
    
    
    
    
    
    
    
    
    
    
    
#        trainNeuron = []
#        trainTimes = []	
#        train_len = 0
#          
#        n_steps = 1
#          
#        for n in range(n_steps):
#              
#            self.reset(rec=False,poi=True)
#              
#            self.measures = self.Data['measureTimes']
#              
#            trainNeuron.append([])
#            trainTimes.append([])
#            last_spike = 0
#              
#            while self.T < self.paras['TC']:
#                dt_bool, next_spike = self.simple_iteration()
#              
#                if dt_bool:		#if next spike is reached before measurement
##                    self.update_spike(next_spike)
#                      
#                    # save spike train
#                    trainNeuron[n]
#                    trainTimes[n].append(self.T)
#                else:
#                    # save phases at measure times
#                    self.Data['phases'][n][self.t-1] = self.phi[:self.paras['N']]
#                    
#    #              if n_steps == 1:
#    #                  print 'total number of spikes: %d, resulting firing rate: %g' % (np.sum(self.s[:self.N]),np.sum(self.s[:self.N])/(t_max*self.N))
          
  
    
  
    def create_topology(self,options):
    
        N_0 = 0
        self.A = np.zeros([self.paras['N']+self.paras['drive_N_p'],self.paras['N']+self.paras['drive_N_p']],dtype=bool)   ## should be sparse matrix (lil_matrix?)
        
        for x in ['E','I']:
            
            if options['A_mode'] == 'uniform':
                for n in range(self.paras['N_%s'%x]):
                    self.A[rnd.sample(range(self.paras['N']),self.paras['K_%s'%x]),N_0+n] = True;
              
            elif options['A_mode'] == 'ER':
            
                self.A[0:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]] = np.random.random((self.paras['N'],self.paras['N_%s'%x])) < self.paras['p_%s'%x]
            
            elif options['A_mode'] == 'lognorm':
                print('not yet available')
                return 0
            else:
                return 0
          
            N_0 += self.paras['N_%s'%x]
          
        np.fill_diagonal(self.A,False)
        
        ### create input layer
#        for n in range(self.paras['N_p']):
#            self.A[N_0+n] = 
        
        ## different distributions for exc and inh? learning only for one?
        self.J = np.zeros([self.paras['N'],self.paras['N']])
        
        N_0 = 0
        for x in ['E','I']:
            if options['J_mode'] == 'uniform':
                self.J[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]] = self.A[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]] * self.paras['J_%s'%x]
                
            elif options['J_mode'] == 'gauss':
                idx_A = self.A[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]] > 0
                numJ = np.sum(idx_A)
                while numJ > 0:
#                    print(idx_A.shape)
#                    print(numJ)
#                    print(np.random.normal(self.paras['J_%s'%x],np.sqrt(self.paras['J_%s'%x]),numJ).shape)
#                    print(self.J[N_0:N_0+self.paras['N_%s'%x],:self.paras['N']][idx_A].shape)
                    print(idx_A.shape)
                    print(numJ)
                    print(self.J[N_0:N_0+self.paras['N_%s'%x],:self.paras['N']][idx_A].shape)
                    print(np.random.normal(self.paras['J_%s'%x],np.sqrt(self.paras['J_%s'%x]),numJ))
                    self.J[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]][idx_A] = np.random.normal(self.paras['J_%s'%x],np.sqrt(self.paras['J_%s'%x]),numJ)    ## shouldnt variance be independent?
                    J_tmp = np.abs(self.J[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]])
                    idx_A = np.logical_or(J_tmp<0,J_tmp>self.paras['STDP_J_max'])
                    numJ = np.sum(idx_A)        
                  
            elif options['J_mode'] == 'lognorm':
                J = np.abs(self.paras['J_%s'%x])
                J_mu = get_logn_mu(J,np.sqrt(J))
                J_var = get_logn_var(J,np.sqrt(J))
                  
                idx_A = self.A[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]] > 0
                numJ = np.sum(idx_A)
                while numJ > 0:
                    self.J[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]][idx_A] = np.sign(self.paras['J_%s'%x])*np.random.lognormal(J_mu,J_var,numJ)
                    
                    idx_A = np.abs(self.J[:self.paras['N'],N_0:N_0+self.paras['N_%s'%x]]) > self.paras['STDP_J_max']
                    numJ = np.sum(idx_A)
            else:
                return 0
          
            N_0 += self.paras['N_%s'%x]
        
        self.delay = np.zeros(self.paras['N'])
        self.delay[:] = np.NaN
        ### allow for delay: add matrix with "interneurons" / "axons", that get activated and spike after tau
        #return 1
  
    
    def test_topology(self):
        
        ### test topology
        Kin_E = np.sum(self.A[:self.paras['N_E'],:],1)
        Kin_I = np.sum(self.A[self.paras['N_E']:,:],1)
        Kout_E = np.sum(self.A[:,:self.paras['N_E']],0)
        Kout_I = np.sum(self.A[:,self.paras['N_E']:],0)
        
        plt.figure(1)
        plt.subplot(2,2,1)
        plt.imshow(self.A,cmap='binary',interpolation='none')
        
        plt.subplot(2,4,3)
        plt.hist(Kin_E,np.linspace(0,(self.paras['K_E']+self.paras['K_I'])*1.5,51),color='k')
        plt.hist(Kin_I,np.linspace(0,(self.paras['K_E']+self.paras['K_I'])*1.5,51),color='r',rwidth=2)
        plt.xlabel('K_{in}')
        
        plt.subplot(2,4,4)
        plt.hist(Kout_E,np.linspace(0,(self.paras['K_E']+self.paras['K_I'])*1.5,51),color='k')
        plt.hist(Kout_I,np.linspace(0,(self.paras['K_E']+self.paras['K_I'])*1.5,51),color='r',rwidth=2)
        plt.xlabel('K_{out}')
        
        plt.subplot(2,2,3)
        J_E = self.J[:,:self.paras['N_E']][self.A[:,:self.paras['N_E']]]
        J_I = self.J[:,self.paras['N_E']:][self.A[:,self.paras['N_E']:]]
        plt.hist(J_E,np.linspace(-5/np.sqrt(self.paras['K']),5/np.sqrt(self.paras['K']),101),color='k')
        plt.hist(J_I,np.linspace(-5/np.sqrt(self.paras['K']),5/np.sqrt(self.paras['K']),101),color='r')
        plt.plot([np.mean(J_E),np.mean(J_E)],[0,0.1*self.paras['K']*self.paras['N']],'k--')
        plt.plot([np.mean(J_I),np.mean(J_I)],[0,0.1*self.paras['K']*self.paras['N']],'r--')
        plt.xlabel('synaptic weights J')
        
        plt.subplot(2,2,4)
        sum_J_E = np.sum(self.J[:self.paras['N_E'],:],1)
        sum_J_I = np.sum(self.J[self.paras['N_E']:,:],1)
        plt.hist(sum_J_E,color='k')
        plt.hist(sum_J_I,color='r')
        plt.xlabel('sum Js')
        
        plt.show(block=False)
	
    #if self.poisson['rate'] and poi:
      #if rec:
        #self.phi = np.append(self.phi,np.zeros(self['N']))
      ##else:
      #for n in range(self['N']):
        #self.phi[self['N']+n] = self.phi_T - self.poisson['train'][n][0]/self.T_free
  
  
  #def generate_poisson_train(self,t_max):
    
    #self.poisson['train'] = []
    
    #for n in range(self['N']):
      #self.poisson['train'].append([])
      #T = 0
      #while T < t_max:
	#spikeTime = -np.log(1-np.random.random())/self.poisson['rate']
	#self.poisson['train'][n].append(T + spikeTime)

	#T += spikeTime
    ##print self.poisson['train']
    
    
    
#    def simple_iteration(self):
#        
#        next_spike = np.argmax(self.phi)
#        dt_spike = max(0,(self.paras['phi_T'] - self.phi[next_spike])*self.T_free)
#        
#        dt_deliver = np.nanmin(self.delay)
#        dt_measure = self.measures[self.t]-self.T
#        
##        print([dt_spike,dt_deliver,dt_measure])
#        dt = np.nanmin([dt_spike,dt_deliver,dt_measure])
#        self.phi += dt / self.T_free
#        self.T += dt			# update current time
#        
#        dt_bool = dt < dt_measure
#        
#        if dt_bool:
#            if np.isnan(dt_deliver) or dt_spike < dt_deliver:
##                print('T=%5.3f: spike at neuron %s'%(self.T,next_spike))
#                self.delay[next_spike] = 10**(-3)+10**(-4)*np.random.normal(1)
#                self.phi[next_spike] = self.paras['phi_R']
#                
#            else:
#                next_deliver = np.where(self.delay==dt_deliver)[0]
##                print('T=%5.3f: spike of neuron %d delivered'%(self.T,next_deliver))
#                self.s[next_deliver] += 1
#                self.phi[self.A[next_deliver][0]] = self.PRC(next_deliver)	# implement J
#                self.delay[next_deliver] = np.NaN
#
#        else:
#            self.t += 1
#        
#        self.delay -= dt
#        
#        return dt_bool, next_spike
  
#    def deliver_spike(self,next_spike):
#        print('update')
#        self.s[next_spike] += 1
#         print "next spike: ", next_spike
        
#        if next_spike < self.paras['N']:
#            self.phi[next_spike] = 0
      #else:
          #next_poisson = next_spike-self['N']
          ##print self.poisson['train'][next_poisson]
          ##print "next poisson spike (%d): %g" % (self.s[next_spike],next_poisson)
          ##print "time: ", self.poisson['train'][next_poisson][self.s[next_spike]]
          #self.phi[next_spike] = self.phi_T - (self.poisson['train'][next_poisson][self.s[next_spike]] - self.T)/self.T_free
      
          #print next_poisson
      
#        self.phi[self.A[next_spike]] = self.PRC(next_spike)	# implement J
    
    
#    def PRC(self,next_spike):
#        return -self.paras['tauM']/self.T_free * np.log(np.exp(-self.phi[self.A[next_spike][0]]*self.T_free/self.paras['tauM']) - self.J[next_spike][self.A[next_spike]]/self.paras['I_ext'])

    
    def set_params(self,paras):
        
        print(paras)
        self.paras = {}
        ### set up basic neuron properties
        self.set_default(paras,'phi_T',1)
        self.set_default(paras,'V_T',1)
        self.set_default(paras,'phi_R',0)
        self.set_default(paras,'V_R',0)
        self.set_default(paras,'tauM',0.01)
        
        ### set up network properties
        self.set_default(paras,'N_E',800)
        self.set_default(paras,'N_I',200)
        self.set_default(paras,'N',self.paras['N_E']+self.paras['N_I'])
        
#        self.set_default(paras,'r_E',1)         # firing rate exc
#        self.set_default(paras,'r_I',1)         # firing rate inh
        
        self.set_default(paras,'p_E',0.1)
        self.set_default(paras,'p_I',0.4)
        
        if self.paras['N_E']:
            self.set_default(paras,'K_E',int(self.paras['p_E']*self.paras['N']))
        else:
            self.set_default(paras,'K_E',0)
        
        if self.paras['N_I']:
            self.set_default(paras,'K_I',int(self.paras['p_I']*self.paras['N']))
        else:
            self.set_default(paras,'K_I',0)
    
        self.paras['K'] = self.paras['p_E']*self.paras['N'] + self.paras['p_I']*self.paras['N']     # overall incoming synapses
        
        self.set_default(paras,'J_E',1/np.sqrt(self.paras['K']))
        self.set_default(paras,'J_I',-1/np.sqrt(self.paras['K']))
        
#        K = self.paras['p_E']*self.paras['N_E'] + self.paras['p_I']*self.paras['N_I']
        #I_ext = - (self.paras['J_E']*self.paras['p_E']*self.paras['N_E']*self.paras['r_E'] + self.paras['J_I']*self.paras['p_I']*self.paras['N_I']*self.paras['r_I'])
        
#        self.set_default(paras,'Iext',I_ext/self.paras['K'])
        self.set_default(paras,'Iext',0)
        
        print('Iext:')
        print(self.paras['Iext']/self.paras['K'])
        
        self.set_default(paras,'gamma',2)
        ### set up simulation properties
        #self.set_default(paras,'TR',2)
        self.set_default(paras,'TW',0.1)      # time for warmup
        self.set_default(paras,'TC',1)     # time for calculation
        
        self.paras['STDP_tau_neg'] = 0.01
        self.paras['STDP_tau_pos'] = 0.01
        self.paras['STDP_eta_neg'] = 0.1
        self.paras['STDP_eta_pos'] = 0.1
        self.paras['STDP_J_max'] = (self.paras['V_T']-self.paras['V_R'])/3
        
        self.set_default(paras,'drive_N_p',int(0.1*self.paras['N']))             ## number of driving neurons
        self.set_default(paras,'drive_r_p',1)                               ## base rate
        self.set_default(paras,'drive_r_p_ampl',5*self.paras['drive_r_p'])  ## maximum rate when at preferential place
        self.set_default(paras,'drive_K_sigm',0.1*self.paras['N_E'])        ## width of gaussian connection probability distribution
        self.set_default(paras,'drive_J',1/np.sqrt(self.paras['K']))        ## connection strengths of input neurons (scaling by K? or overall K? should be distribution?)
        
        self.set_default(paras,'N_PC',0.5*self.paras['N_E'])
        
#        self.T_free = -self.paras['tauM']*np.log((self.paras['V_T'] - self.paras['Iext'])/(self.paras['V_R'] - self.paras['Iext']))
        
        print(self.paras)
        
  
    def set_default(self,paras,key,defVal):
        
        if not(key in paras):
            self.paras[key] = defVal
        else:
            self.paras[key] = paras[key]



net = network({'N_I':100,'N_E':400,'drive_N_p':0,'p_E':0.1,'p_I':0.4,'Iext':0.0,'TC':100},{'A_mode':'ER','J_mode':'lognorm','test':True})

net.simulate()

#net.test_topology()

#plt.figure(2)
#plt.scatter(net.Data['spikeTrain'][1],net.Data['spikeTrain'][0],marker='|')
#plt.show()
#
#plt.ion()
#
#fig = plt.figure(3)
#ax1 = fig.add_subplot(211)
#p_hist, = ax1.plot(np.linspace(-net.paras['STDP_J_max'],net.paras['STDP_J_max'],100),net.Data['J_hist'][0])
#
#ax2 = fig.add_subplot(212)
#p_diff, = ax2.plot(np.linspace(-net.paras['STDP_J_max'],net.paras['STDP_J_max'],100),net.Data['J_hist'][0]-net.Data['J_hist'][0])
##plt.show(block=False)
#
#### should be a video
##while True:
#for i in range(len(net.Data['J_hist'])):
##    plt.subplot(3,int(np.ceil(len(net.Data['J_hist'])/3)),i+1)
#    ax1.clear()
#    ax1.plot(np.linspace(-net.paras['STDP_J_max'],net.paras['STDP_J_max'],100),net.Data['J_hist'][i])
#    ax2.clear()
#    ax2.plot(np.linspace(-net.paras['STDP_J_max'],net.paras['STDP_J_max'],100),net.Data['J_hist'][i]-net.Data['J_hist'][0])
##    plt.plot(np.linspace(-net.paras['STDP_J_max'],net.paras['STDP_J_max'],100),net.Data['J_hist'][i])
##        p_hist.set_ydata(net.Data['J_hist'][i])
##        p_diff.set_ydata(net.Data['J_hist'][i]-net.Data['J_hist'][0])
#    
##        fig.canvas.flush_events()
#    ax1.set_title("t=%d"%i)
#    plt.pause(0.2)
##        plt.show(block=False)
##        fig.canvas.draw()
##        plt.canvas.draw()
##        time.sleep(1)
##plt.show()
