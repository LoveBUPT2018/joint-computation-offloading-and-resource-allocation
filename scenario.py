# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:49 2020

@author: liangyu

Create the network simulation scenario
"""

import numpy as np
from numpy import pi
from random import random, uniform, choice

class MEC:  # Define the MEC
    
    def __init__(self, sce, MEC_index, MEC_type, MEC_Loc, MEC_Radius):
        self.sce = sce
        self.id = MEC_index
        self.MECtype = MEC_type
        self.MEC_Loc = MEC_Loc
        self.MEC_Radius = MEC_Radius
        
    def reset(self):  # Reset the RB status
        self.Ch_State = np.zeros(self.sce.nRB)    
        
    def Get_Location(self):
        return self.MEC_Loc
    
    def Transmit_Power_dBm(self):  # Calculate the transmit power of a MEC
        if self.MECtype == "MMEC":
            Tx_Power_dBm = 40   
        elif self.MECtype == "PMEC":
            Tx_Power_dBm = 30 
        elif self.MECtype == "FMEC":
            Tx_Power_dBm = 20 
        return Tx_Power_dBm  # Transmit power in dBm, no consideration of power allocation now
    
    def Receive_Power(self, d):  # Calculate the received power by transmit power and path loss of a certain MEC
        Tx_Power_dBm = self.Transmit_Power_dBm()
        if self.MECtype == "MMEC" or self.MECtype == "PMEC":
            loss = 34 + 40 * np.log10(d)
        elif self.MECtype == "FMEC":
            loss = 37 + 30 * np.log10(d)  
        if d <= self.MEC_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss  # Received power in dBm
            Rx_power = 10**(Rx_power_dBm/10)  # Received power in mW
        else:
            Rx_power = 0.0
        return Rx_power        
        
        
class Scenario:  # Define the network scenario
    
    def __init__(self, sce):  # Initialize the scenario we simulate
        self.sce = sce
        self.MECs = self.MEC_Init()
        
    def reset(self):   # Reset the scenario we simulate
        for i in range(len(self.MECs)):
            self.MECs[i].reset()
            
    def MEC_Number(self):
        nMEC = self.sce.nMMEC + self.sce.nPMEC + self.sce.nFMEC  # The number of MECs
        return nMEC
    
    def MEC_Location(self):
        Loc_MMEC = np.zeros((self.sce.nMMEC,2))  # Initialize the locations of MECs
        Loc_PMEC = np.zeros((self.sce.nPMEC,2))
        Loc_FMEC = np.zeros((self.sce.nFMEC,2)) 
        
        for i in range(self.sce.nMMEC):
            Loc_MMEC[i,0] = 500 + 900*i  # x-coordinate
            Loc_MMEC[i,1] = 500  # y-coordinate
        
        for i in range(self.sce.nPMEC):
            Loc_PMEC[i,0] = Loc_MMEC[int(i/4),0] + 250*np.cos(pi/2*(i%4))
            Loc_PMEC[i,1] = Loc_MMEC[int(i/4),1] + 250*np.sin(pi/2*(i%4))
            
        for i in range(self.sce.nFMEC):
            LocM = choice(Loc_MMEC)
            r = self.sce.rMMEC*random()
            theta = uniform(-pi,pi)
            Loc_FMEC[i,0] = LocM[0] + r*np.cos(theta)
            Loc_FMEC[i,1] = LocM[1] + r*np.sin(theta)

        return Loc_MMEC, Loc_PMEC, Loc_FMEC
    
    def MEC_Init(self):   # Initialize all the MECs
        MECs = []  # The vector of MECs
        Loc_MMEC, Loc_PMEC, Loc_FMEC = self.MEC_Location()
        
        for i in range(self.sce.nMMEC):  # Initialize the MMECs
            MEC_index = i
            MEC_type = "MMEC"
            MEC_Loc = Loc_MMEC[i]
            MEC_Radius = self.sce.rMMEC            
            MECs.append(MEC(self.sce, MEC_index, MEC_type, MEC_Loc, MEC_Radius))
            
        for i in range(self.sce.nPMEC):
            MEC_index = self.sce.nMMEC + i
            MEC_type = "PMEC"
            MEC_Loc = Loc_PMEC[i]
            MEC_Radius = self.sce.rPMEC
            MECs.append(MEC(self.sce, MEC_index, MEC_type, MEC_Loc, MEC_Radius))
            
        for i in range(self.sce.nFMEC):
            MEC_index = self.sce.nMMEC + self.sce.nPMEC + i
            MEC_type = "FMEC"
            MEC_Loc = Loc_FMEC[i]
            MEC_Radius = self.sce.rFMEC
            MECs.append(MEC(self.sce, MEC_index, MEC_type, MEC_Loc, MEC_Radius))
        return MECs
            
    def MECs(self):
        return self.MECs


        
            
    

