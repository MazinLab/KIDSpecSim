# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:34:52 2020

@author: BVH
"""

'''
This code produces a photon time stream array, with array of time and photon count at that time.
NOTE: In the final 'photon_stream' array, the counts are done so that the photon's wavelength has been added to the array. So 
the input for a 500nm photon incoming would have been to, for the current time bin, add 500.
Whereas the array 'count_stream' has the counts of each order and overall. 
'''

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

plt.rcParams.update({'font.size': 32})

#setting up

#'data' to be used
#spec = np.load('HD212442_photon_spec.npy')
#spec = np.load('HD212442_photon_spec_with_sky.npy')
#wl = mu = spec[0][1]
#tot_count = spec[1][1]

exposure_t = 50

def nearest(x,value,val_or_coord):
    coord = np.abs(x-value).argmin()
    val = x[coord]
    if val_or_coord == 'coord':
        return coord
    if val_or_coord == 'val':
        return val
    
def nearest_3(x,value,coord_removal_val):
    x_new = np.zeros_like(x)
    x_new += x
    out = np.zeros(3)
    
    coord = np.abs(x_new-value).argmin()
    out[0] = coord
    
    x_new[coord] += coord_removal_val
    coord = np.abs(x_new-value).argmin()
    out[1] = coord
    
    x_new[coord] += coord_removal_val
    coord = np.abs(x_new-value).argmin()
    out[2] = coord
 
    return out


def photon_stream_generator(R,plotting=False):

    data = np.zeros((3,3))
    data[:,0] = np.array([8,9,10])
    data[:,1] = np.array([550.14,489.01,440.11])
    data[:,2] = np.array([2000,2000*100,2000])
    
    #data = np.zeros((3,3))
    #data[:,0] = np.array([9,10,11])
    #data[:,1] = np.array([698.74,628.87,571.70])
    #data[:,2] = np.array([16983.5,17457.5,17612.3])
    
    len_ord = int(len(data[:,0]))
    
    exp_t = 50 #in seconds
    time_res = 1/(1e6) #1 microsecond time res assumed
    
    time_steps = (exp_t / time_res) /100
    
    photon_stream = np.zeros((2+len_ord,int(time_steps)))
    photon_stream[0] = np.linspace(0,time_steps,int(time_steps))
    
    count_stream = np.zeros_like(photon_stream)
    count_stream[0] = photon_stream[0]
    
    all_gauss_store = []
    all_poisson_store = []
    gauss_features = np.zeros((len_ord,2))
    
    
    
    #Starting photon stream simulation
    for k in range(len_ord):
        
        mu = data[k,1]
        tot_count = data[k,2]
        
        expect_count = tot_count/time_steps
        
        #MKID gaussian
        KID_R =  R / np.sqrt(mu/400)
        
        sig = mu / (KID_R * (2*np.sqrt( 2*np.log(2) )))  #using an equation from 'GIGA-Z: A 100,000 OBJECT SUPERCONDUCTING SPECTROPHOTOMETER FOR LSST FOLLOW-UP' 
        
        gauss_features[k,:] = np.array([mu,sig])
        
        gaussian_eff_store = []
        poisson_eff_store = []
        
        for i in range(int(time_steps)):
            poisson_eff = np.random.choice(2,p=[1-expect_count,expect_count])
            gaussian_wl = np.random.normal(loc=mu,scale=sig,size=poisson_eff)
            gaussian_eff_store.append(gaussian_wl)
            poisson_eff_store.append(poisson_eff)
            if poisson_eff > 0:
                for j in range(poisson_eff):
                    photon_stream[k+1][i] += (gaussian_wl[j])
                    count_stream[k+1][i] += (mu/gaussian_wl[j])
            prog = ((i+1) / int(time_steps))*100
            print('\r%.5f%% of timebins in %ith of %i orders complete.'%(prog,k+1,len_ord),end='',flush=True)
                
            
        #print('\nSpectrum count:',tot_count)
        #print('\nPhoton stream count:',np.sum(count_stream[k+1]))
        #print('\nDifference:', abs(tot_count-np.sum(count_stream[k+1])),'\n')
        
        all_gauss_store.append(gaussian_eff_store)
        all_poisson_store.append(poisson_eff_store)
    print('\n')
    for i in range(int(time_steps)):
        photon_stream[-1,i] = np.sum(photon_stream[1:,i])
        count_stream[-1,i] = np.sum(count_stream[1:,i])
    
        print('\r%i/%i time bins summed.'%(i+1,time_steps),end='',flush=True)
    
    
    if plotting == True:
        colors = ['r','g','b']
        #PLOTTING
        plt.figure()
        plt.title('Total Photon Stream')
        plt.plot(photon_stream[0],photon_stream[-1],'r-')
        plt.xlabel('Time / $\mu$s')
        plt.ylabel('Summed wavelengths detected / nm')
        
        plt.figure()
        plt.title('Photon Poisson Distribution')
        for i in range(len_ord):
            if np.max(all_poisson_store[i]) > 0:
                plt.hist(all_poisson_store[i],histtype='step',bins=int(np.max(all_poisson_store[i])),label='Ord. %i'%data[i,0])
        plt.legend(loc='best')
        plt.xlabel('Photon Counts per Time Bin')
        plt.ylabel('Frequency')
        
        fig3 = plt.figure()
        #plt.title('MKID Gaussian')
        fig3.text(0.78,0.8,'$R_E$=%.1f'%R)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Frequency')
        plt.xlim(350,700)
        outputs = []
        for j in range(len_ord):
            gaussian_set = all_gauss_store[j]
            
            gaussians = gaussian_set[0]
            
            for i in range(len(gaussian_eff_store)-1):
                gaussians = np.concatenate((gaussians,gaussian_set[i+1]))
                
            plt.hist(gaussians,bins=60,histtype='barstacked',color=colors[j],edgecolor='k',label='Ord. %i'%data[j,0])
            outputs.append( np.histogram(gaussians,bins=30))
            
        
        plt.legend(loc='upper left')
        
        for j in range(len_ord):
            gaussian_set = all_gauss_store[j]
            
            gaussians = gaussian_set[0]
            
            for i in range(len(gaussian_eff_store)-1):
                gaussians = np.concatenate((gaussians,gaussian_set[i+1]))
                
            plt.hist(gaussians,bins=60,histtype='step',linewidth=1.1,color='k')
            
        
    
    return gauss_features,photon_stream,count_stream,data,outputs

#####################################################################################################################################################################################
    
def photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False):
    
    data = np.zeros((len(w_o),3))
    data[:,0] = orders
    data[:,1] = w_o
    data[:,2] = pixel_sum
    
    len_ord = int(len(data[:,0]))

    time_res = 1/1000 #1 microsecond time res assumed
    
    time_steps = exposure_t / time_res
    
    photon_stream = np.zeros((2+len_ord,int(time_steps)))
    photon_stream[0] = np.linspace(0,time_steps,int(time_steps))
    
    count_stream = np.zeros_like(photon_stream)
    count_stream[0] = photon_stream[0]
    
    all_gauss_store = []
    all_poisson_store = []
    gauss_features = np.zeros((len_ord,2))
    
    
    
    #Starting photon stream simulation
    for k in range(len_ord):
        
        mu = data[k,1]
        tot_count = data[k,2]
        
        expect_count = tot_count/time_steps
        
        #MKID gaussian
        KID_R =  R / np.sqrt(mu/400)
        
        sig = mu / (KID_R * (2*np.sqrt( 2*np.log(2) )))  #using an equation from 'GIGA-Z: A 100,000 OBJECT SUPERCONDUCTING SPECTROPHOTOMETER FOR LSST FOLLOW-UP' 
        
        gauss_features[k,:] = np.array([mu,sig])
        
        gaussian_eff_store = []
        poisson_eff_store = []
        
        for i in range(int(time_steps)):
            poisson_eff = np.random.poisson(lam=expect_count)
            gaussian_wl = np.random.normal(loc=mu,scale=sig,size=poisson_eff)
            gaussian_eff_store.append(gaussian_wl)
            poisson_eff_store.append(poisson_eff)
            if poisson_eff > 0:
                for j in range(poisson_eff):
                    photon_stream[k+1][i] += (gaussian_wl[j])
                    count_stream[k+1][i] += poisson_eff
                
            
        #print('\nSpectrum count:',tot_count)
        #print('\nPhoton stream count:',np.sum(count_stream[k+1]))
        #print('\nDifference:', abs(tot_count-np.sum(count_stream[k+1])),'\n')
        
        all_gauss_store.append(gaussian_eff_store)
        all_poisson_store.append(poisson_eff_store)
    
    for i in range(int(time_steps)):
        photon_stream[-1,i] = np.sum(photon_stream[1:,i])
        count_stream[-1,i] = np.sum(count_stream[1:,i])
    
    count_sum = np.zeros((len(count_stream[1:-1,0]),2))
    count_sum[:,0] = gauss_features[:,0]
    for i in range(len(count_stream[1:-1,0])):
        count_sum[i,1] = np.sum(count_stream[i+1])
    
    
    if plotting == True:
        #PLOTTING
        plt.figure()
        plt.title('Total Photon Stream')
        plt.plot(photon_stream[0],photon_stream[-1],'r-')
        plt.xlabel('Time / $\mu$s')
        plt.ylabel('Summed wavelengths detected / nm')
        
        plt.figure()
        plt.title('Photon Poisson Distribution')
        for i in range(len_ord):
            if np.max(all_poisson_store[i]) > 0:
                plt.hist(all_poisson_store[i],histtype='step',bins=int(np.max(all_poisson_store[i])),label='Ord. %i'%data[i,0])
        plt.legend(loc='best')
        plt.xlabel('Photon Counts per Time Bin')
        plt.ylabel('Frequency')
        
        fig3 = plt.figure()
        plt.title('MKID Gaussian')
        fig3.text(0.78,0.8,'$R_E$=%.1f'%R)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Frequency')
        
        for j in range(len_ord):
            gaussian_set = all_gauss_store[j]
            
            gaussians = gaussian_set[0]
            
            for i in range(len(gaussian_eff_store)-1):
                gaussians = np.concatenate((gaussians,gaussian_set[i+1]))
                
            plt.hist(gaussians,bins=30,histtype='barstacked',edgecolor='k',label='Ord. %i'%data[j,0])
        
        plt.legend(loc='upper left')
        
        for j in range(len_ord):
            gaussian_set = all_gauss_store[j]
            
            gaussians = gaussian_set[0]
            
            for i in range(len(gaussian_eff_store)-1):
                gaussians = np.concatenate((gaussians,gaussian_set[i+1]))
                
            plt.hist(gaussians,bins=30,histtype='step',linewidth=1.1,color='k')
        
        plt.tight_layout()
    
    return gauss_features,photon_stream,count_stream,data,count_sum

####################################################################################################################################################################################

def photon_assigner(gauss_features,photon_stream):
    
    photon_result = np.zeros((len(gauss_features[:,0])+1,len(photon_stream[0])))
    photon_result[0] = photon_stream[0]
    
    for i in range(len(photon_stream[0])):
        
        if photon_stream[-1,i] != 0:
            probs = np.zeros(len(gauss_features[:,0]))
            for j in range(len(gauss_features[:,0])):
                gauss_prob = scipy.stats.norm(gauss_features[j,0],gauss_features[j,1]) #Needs a zero condition
                probs[j] = gauss_prob.pdf(photon_stream[-1,i])
            
            max_prob = probs.argmax()
            photon_result[max_prob+1,i] = gauss_features[max_prob,0]
        else:
            photon_result[1:,i] = 0
    
    return photon_result

######################################################################################################################################################################################

def photon_result_processor(photon_result,count_stream,gauss_features):
    
    pix_order_ph = np.zeros((2,len(gauss_features[:,0])))
    pix_order_mis = np.zeros((2,len(gauss_features[:,0])))
    
    pix_order_ph[0] = gauss_features[:,0]
    pix_order_mis[0] = gauss_features[:,0]
    
    for i in range(len(gauss_features[:,0])):
        count_ord = np.sum(photon_result[i+1])/gauss_features[i,0]
        count_diff = np.abs(np.sum(count_stream[i+1])-count_ord)
        
        pix_order_ph[1,i] = count_ord
        pix_order_mis[1,i] = count_diff
    
    return pix_order_ph,pix_order_mis
        
####################################################################################################################################################################################### 
        
def combo_calc(gauss_features):
    
    pot_resp = []
    pot_combos = []
    
    len_ord = len(gauss_features[:,0])
    
    key = gauss_features[:,0]
    
    for i in range(len_ord):
        pot_resp.append(gauss_features[i,0])
        pot_combos.append(str(i))
    
    
    #looping for large numbers of combinations
    
    
    #First stage combinations
    combo_1 = np.zeros((len_ord,len_ord))
    for i in range(len_ord):
        for j in range(len_ord):                     
            combo_1[i,j] = key[i] + key[j]
            
    combo_1_1D = combo_1[:,0]
    for i in range(len(combo_1[:,0])):
        cur_coord = str(0)+'+'+str(i)
        pot_combos.append(cur_coord)
        
    i_count = 1
    for i in range(len_ord-1):
        sec = combo_1[i_count:,i+1]
        combo_1_1D = np.concatenate((combo_1_1D,combo_1[i_count:,i+1]))
        i_count += 1
        for j in range(len(sec)):
            cur_coord = str(i+1)+'+'+str(j+(len_ord-len(sec)))
            pot_combos.append(cur_coord)
    
    key2 = np.concatenate((key,combo_1_1D))
    
    #calculating combinations of three incoming photons-------------------------------------------- 
    for i in range(len(combo_1_1D)):
        cur_combo = pot_combos[int(len_ord+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([key2[int(len_ord+i)] +key2[j]])
            
            pot_combos.append(combo_ij)
            key2 = np.concatenate((key2,combo_key_ij))
    #-----------------------------------------------------------------------------------------------
    
    #Second stage combinations
    combo_2 = np.zeros((len(combo_1_1D),len(combo_1_1D)))
    for i in range(len(combo_1_1D)):
        for j in range(len(combo_1_1D)):
            combo_2[i,j] = combo_1_1D[i] + combo_1_1D[j] 
            
    combo_2_1D = combo_2[:,0]
    pot_combos_sec = pot_combos[len(key):len(key)+len(combo_1_1D)]
    for i in range(len(combo_2[:,0])):
        cur_coord = pot_combos_sec[i] + '+' + pot_combos_sec[0]
        pot_combos.append(cur_coord)
        
    i_count = 1
    for i in range(len(combo_1_1D)-1):
        sec = combo_2[i_count:,i+1]
        combo_2_1D = np.concatenate((combo_2_1D,combo_2[i_count:,i+1]))
        i_count += 1
        for j in range(len(sec)):
            cur_coord = pot_combos_sec[j+(len(combo_1_1D)-len(sec))] + '+' + pot_combos_sec[i+1]
            pot_combos.append(cur_coord)
            
    key3 = np.concatenate((key2,combo_2_1D))
    
    #calculating combinations of 5-7 incoming photons--------------------------------------------------
    
    #5
    track = 0
    for i in range(len(combo_2_1D)):
        cur_combo = pot_combos[int(len(key2)+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([ key3[int(len(key2)+i)] + key[j] ])
            
            pot_combos.append(combo_ij)
            key3 = np.concatenate((key3,combo_key_ij))
            track += 1
    
    #6
    track1 = track
    for i in range(track1):
        cur_combo = pot_combos[int(len(key3)-track+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([ key3[int(len(key3)-track+i)] + key[j] ])
            
            pot_combos.append(combo_ij)
            key3 = np.concatenate((key3,combo_key_ij))
            track += 1
    
    #7
    key3_track = key3
    track2 = track
    track_diff = track2-track1
    for i in range(track_diff):
        cur_combo = pot_combos[int(len(key3_track)-track_diff+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([ key3_track[int(len(key3_track)-track_diff+i)] + key[j] ])
            
            pot_combos.append(combo_ij)
            key3 = np.concatenate((key3,combo_key_ij))
            track += 1  
            
     
    #----------------------------------------------------------------------------------------
    
    #Third stage combinations
    combo_3 = np.zeros((len(combo_2_1D),len(combo_2_1D)))
    for i in range(len(combo_2_1D)):
        for j in range(len(combo_2_1D)):
            combo_3[i,j] = combo_2_1D[i] + combo_2_1D[j]   
            
    combo_3_1D = combo_3[:,0]
    pot_combos_sec = pot_combos[len(key2):len(combo_2_1D)+len(key2)]
    for i in range(len(combo_3[:,0])):
        cur_coord = pot_combos_sec[i] + '+' + pot_combos_sec[0]
        pot_combos.append(cur_coord)
        
    i_count = 1
    for i in range(len(combo_2_1D)-1):
        sec = combo_3[i_count:,i+1]
        combo_3_1D = np.concatenate((combo_3_1D,combo_3[i_count:,i+1]))
        i_count += 1
        for j in range(len(sec)):
            cur_coord = pot_combos_sec[j+(len(combo_2_1D)-len(sec))] + '+' + pot_combos_sec[i+1]
            pot_combos.append(cur_coord)
            
    key4 = np.concatenate((key3,combo_3_1D))
        
          
    
    #calculating combinations of 9 and 10 incoming photons--------------------------------------------------            
    
    #9
    key4_track = key4
    track = 0
    for i in range(len(combo_3_1D)):
        cur_combo = pot_combos[int(len(key3)+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([ key4_track[int(len(key3)+i)] + key[j] ])
            
            pot_combos.append(combo_ij)
            key4 = np.concatenate((key4,combo_key_ij))
            track += 1
    
    #10
    track1 = track
    key4_track = key4
    for i in range(track1):
        cur_combo = pot_combos[int(len(key4)-track+i)]
        for j in range(len_ord):
            combo_ij = cur_combo + '+' + pot_combos[j]
            combo_key_ij = np.array([ key4[int(len(key4)-track+i)] + key[j] ])
            
            pot_combos.append(combo_ij)
            key4 = np.concatenate((key4,combo_key_ij))
            track += 1      

    return pot_combos,key4

############################################################################################################################################################################################################                 

def gauss_prob(gauss_features,photon,rec_photon_sec,mult_photon=False,prob_return=False):
    
    if mult_photon == False:
        probs = np.zeros(len(gauss_features[:,0]))
        
        for i in range(len(gauss_features[:,0])):
            gauss_prob = scipy.stats.norm(gauss_features[i,0],gauss_features[i,1])
            probs[i] = gauss_prob.pdf(photon)
            
        max_prob = np.argmax(probs)
        
        photon_wl = gauss_features[max_prob,0]
        
        if prob_return == True:
            return photon_wl,probs
        else:
            return photon_wl
    
    if mult_photon == True:
        
        new_mu = np.zeros(len(gauss_features[:,0]))
        new_sig = np.zeros_like(new_mu)
        for i in range(len(gauss_features[:,0])):
            if rec_photon_sec[i+1] != 0:
                no_photon_ord = int(rec_photon_sec[i+1] / gauss_features[i,0])
                if no_photon_ord > 1:
                    new_mu[i] = gauss_features[i,0]*no_photon_ord
                    new_sig[i] = np.sqrt(no_photon_ord*(gauss_features[i,1]**2))
                else:
                    new_mu[i] = gauss_features[i,0]
                    new_sig[i] = gauss_features[i,1]
        
        final_mu = np.sum(new_mu)
        final_sig = np.sqrt(np.sum(new_sig**2))
        
        gauss_prob = scipy.stats.norm(final_mu,final_sig)
        prob = gauss_prob.pdf(photon)
        
        return prob
    
###################################################################################################################################################################################
            

def photon_stream_recreation(photon_stream,count_stream,data,photon_resp,photon_combos,gauss_features):                  
    recreated_photon_stream = np.zeros_like(photon_stream)        
    recreated_photon_stream[0] = photon_stream[0]
    
    recreated_data = np.zeros_like(data)
    recreated_data[:,0] = data[:,0]
    recreated_data[:,1] = data[:,1]
                    
    for i in range(len(recreated_photon_stream[0])):
        
        current_resp = photon_stream[-1,i]
        
        if current_resp != 0:
        
            resp_coord = nearest(photon_resp,current_resp,'coord')
            
            resp_combo = photon_combos[resp_coord].split('+')
            resp_combo = np.array([int(i) for i in resp_combo])
            
            for j in range(len(resp_combo)):
                
                recreated_photon_stream[resp_combo[j]+1,i] += gauss_features[resp_combo[j],0]
                recreated_data[resp_combo[j],2] += 1
        
        recreated_photon_stream[-1,i] = np.sum(recreated_photon_stream[1:,i])
    
    
    perc_mis = np.zeros((len(data[:,0]),6))
    perc_mis[:,0] = data[:,0]
    perc_mis[:,1] = data[:,1]
    perc_mis[:,2] = data[:,2]
    perc_mis[:,3] = np.sum(count_stream[1:-1],axis=1)
    perc_mis[:,4] = recreated_data[:,2]
    
    
    
    for i in range(len(data[:,0])):
        perc_mis[i,5] = (abs(perc_mis[i,3]-perc_mis[i,4]) / perc_mis[i,3]) * 100
                    
    return perc_mis,recreated_photon_stream,recreated_data


def photon_stream_recreation_V2(photon_stream,count_stream,data,photon_resp,photon_combos,gauss_features):                  
    recreated_photon_stream = np.zeros_like(photon_stream) 
    recreated_count_stream = np.zeros_like(photon_stream) 
    recreated_count_stream[0] = photon_stream[0]      
    recreated_photon_stream[0] = photon_stream[0]
    
    recreated_data = np.zeros_like(data)
    recreated_data[:,0] = data[:,0]
    recreated_data[:,1] = data[:,1]
                    
    for i in range(len(recreated_photon_stream[0])):
        
        current_resp = photon_stream[-1,i]
        
        current_set = np.zeros((len(gauss_features[:,0])+2,3))
        current_set[0,:] = photon_stream[0,i]
        
        if current_resp != 0:
        
            resp_coords = nearest_3(photon_resp,current_resp,10000)
            
            probs = np.zeros(3)
            
            for k in range(len(resp_coords)):
                
                resp_combo = photon_combos[int(resp_coords[k])].split('+')
                resp_combo = np.array([int(i) for i in resp_combo])
                
                for j in range(len(resp_combo)):
                    current_set[resp_combo[j]+1,k] += gauss_features[resp_combo[j],0]
             
                current_set[-1,k] = np.sum(current_set[1:,k])
            
                probs[k] = gauss_prob(gauss_features,current_resp,current_set[:,k],mult_photon=True,prob_return=False)
        
            max_prob = probs.argmax()
        
        else:
            max_prob = 0
        
        for k in range(len(gauss_features[:,0])):
            recreated_data[k,2] += int(current_set[k+1,max_prob] / gauss_features[k,0])
            recreated_count_stream[k+1,i] = int(current_set[k+1,max_prob] / gauss_features[k,0])
            
        recreated_photon_stream[:,i] = current_set[:,max_prob]
        recreated_count_stream[-1,i] = np.sum(recreated_count_stream[1:,i])
    
    perc_mis = np.zeros((len(data[:,0]),6))
    perc_mis[:,0] = data[:,0]
    perc_mis[:,1] = data[:,1]
    perc_mis[:,2] = data[:,2]
    perc_mis[:,3] = np.sum(count_stream[1:-1],axis=1)
    perc_mis[:,4] = recreated_data[:,2]
    
    
    
    for i in range(len(data[:,0])):
        perc_mis[i,5] = (abs(data[i,2]-recreated_data[i,2]) / data[i,2]) * 100
                    
    return perc_mis,recreated_photon_stream,recreated_data,recreated_count_stream


def wave_to_phase(photon_stream):
    
    eta = 0.57
    Fano = 0.5
    Tc = 0.4
    kb = 8.617e-5
    planck = 1.2398*1000.0
    gap = 3.5*kb*Tc/2.0
    ref_wave = 400.0
    #sigma_qp = np.sqrt(Fano*eta*(planck/ref_wave)/gap)
    sigma_excess = 10.
    #sigma_amp = 10.
    qp_per_ev = Fano*eta/gap/sigma_excess
    qp_per_deg = qp_per_ev*(planck/ref_wave)/90.0
    
    phase_time = np.array([[]])
    
    for i in range(len(photon_stream[-1,:])):
        if photon_stream[-1,i] > 0:
            pulse_i = (planck/photon_stream[-1,i])*qp_per_ev
            phase_i = np.real(pulse_i)/qp_per_deg
            #phase_time = np.append(phase_time,np.array([[i,phase_i]]))
            
            if len(phase_time[0]) == 0:
                phase_time = np.append(phase_time,np.array([[photon_stream[0,i],phase_i]]),axis=1)
            else:
                phase_time = np.append(phase_time,np.array([[photon_stream[0,i],phase_i]]),axis=0)
    
    return phase_time
################################################################################################################################################################################



R = np.array([30])#np.linspace(5,85,20)

ord_err_perc = np.zeros((3,len(R)))
ord_err_perc2 = np.zeros((3,len(R)))

prog = 0

for i in range(len(R)):
    print('Starting photon stream generation')
    gauss_features,photon_stream,count_stream,data,out = photon_stream_generator(R[i],plotting=True)
    print('Photon response calculation')
    photon_combos,photon_resp = combo_calc(gauss_features) 
    
    print('Processing 1')
    perc_mis2,recreated_photon_stream2,recreated_data2,recreated_count_stream2 = photon_stream_recreation_V2(photon_stream,count_stream,data,photon_resp,photon_combos,gauss_features)
    print('Processing 2')
    perc_mis,recreated_photon_stream,recreated_data = photon_stream_recreation(photon_stream,count_stream,data,photon_resp,photon_combos,gauss_features)
    
    ord_err_perc[:,i] = perc_mis[:,5]
    ord_err_perc2[:,i] = perc_mis2[:,5]
    
    prog += 1
    
    print('\n',np.round((prog/len(R))*100,decimals=2),'% complete')
    

wl_bins = []
counts = []
for i in range(3):
    wl_bins.append(out[i][1])
    counts.append(out[i][0])
wl_overlap = np.zeros(2)
total_photons = np.sum(counts[:])
counts_overlap = np.zeros((2,2))
for i in range(2):
    coord_1 = np.where(wl_bins[i][0] >= wl_bins[i+1])[0][-1]
    print(np.where(wl_bins[i+1][-1] >= wl_bins[i]))
    coord_2 = np.where(wl_bins[i+1][-1] >= wl_bins[i])[0][-1]
    #print(coord_1)
    #print(coord_2)
    wl_overlap[i] +=  wl_bins[i+1][-1] - wl_bins[i+1][coord_1]
    counts_overlap[i,0] += np.sum(counts[i][:coord_2])
    counts_overlap[i,1] += np.sum(counts[i+1][coord_1:])

sum_counts_overlap = np.sum(counts_overlap)
perc_overlap = (sum_counts_overlap/total_photons)*100
print(perc_overlap)

'''
plt.figure()
plt.plot(R,ord_err_counts[0],'b-',markersize=2,label='Order %i - %.2fnm with prob.'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts[0],'bv',markersize=3)       
plt.plot(R,ord_err_counts2[0],'r-',markersize=2,label='Order %i - %.2fnm'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts2[0],'rv',markersize=3)   
plt.xlabel('Energy Resolution / $R_E$')
plt.ylabel('Error / %')
plt.legend(loc='best')
plt.grid()

plt.figure()
plt.plot(R,ord_err_counts[1],'g-',markersize=2,label='Order %i - %.2fnm with prob.'%(data[1,0],data[1,1]))
plt.plot(R,ord_err_counts[1],'gv',markersize=3) 
plt.plot(R,ord_err_counts2[1],'r-',markersize=2,label='Order %i - %.2fnm'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts2[1],'rv',markersize=3)
plt.xlabel('Energy Resolution / $R_E$')
plt.ylabel('Error / %')
plt.legend(loc='best')
plt.grid()

plt.figure()
plt.plot(R,ord_err_counts[2],'m-',markersize=2,label='Order %i - %.2fnm with prob.'%(data[2,0],data[2,1]))
plt.plot(R,ord_err_counts[2],'mv',markersize=3)   
plt.plot(R,ord_err_counts2[2],'r-',markersize=2,label='Order %i - %.2fnm'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts2[2],'rv',markersize=3)
plt.xlabel('Energy Resolution / $R_E$')
plt.ylabel('Error / %')
plt.legend(loc='best')
plt.grid()   

plt.figure()
plt.plot(R,ord_err_counts[3],'k-',markersize=2,label='Order %i - %.2fnm'%(data[3,0],data[3,1]))
plt.plot(R,ord_err_counts[3],'kv',markersize=3) 
plt.plot(R,ord_err_counts2[2],'r-',markersize=2,label='Order %i - %.2fnm'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts2[2],'rv',markersize=3)          
plt.xlabel('Energy Resolution / $R_E$')
plt.ylabel('Error / %')
plt.legend(loc='best')
plt.grid()   

'''


'''    
plt.figure()
plt.plot(R,ord_err_counts[0],'b-',markersize=2,label='Order %i - %.2fnm'%(data[0,0],data[0,1]))
plt.plot(R,ord_err_counts[0],'bv',markersize=3)       

plt.figure()
plt.plot(R,ord_err_counts[1],'g-',markersize=2,label='Order %i - %.2fnm'%(data[1,0],data[1,1]))
plt.plot(R,ord_err_counts[1],'gv',markersize=3) 

plt.figure()
plt.plot(R,ord_err_counts[2],'r-',markersize=2,label='Order %i - %.2fnm'%(data[2,0],data[2,1]))
plt.plot(R,ord_err_counts[2],'rv',markersize=3)      

#plt.plot(R,ord_err_counts[3],'m-',markersize=2,label='Order %i - %.2fnm'%(data[3,0],data[3,1]))
#plt.plot(R,ord_err_counts[3],'mv',markersize=3)           
        
plt.xlabel('Energy Resolution / $R_E$')
plt.ylabel('Error / %')
plt.legend(loc='best')
plt.grid()
'''       
        


