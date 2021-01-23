#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:27:52 2021

@author: krishna.kottakki
"""

import pandas as pd
from pulp import *
import time
import numpy as np
import copy as cp

np.random.seed(0) 


# In[READ ME SECTION] 
print ('Please read the assumptions before executing and analysing \
               solution of this file ...\n') 

# ====   The following assumptions are made and the code will strictcly 
#        restricted to the same. 

# ==== (1) ROWS, LEVELS and GROUPS are considered to representing the same information 
# ==== (2) GRIDS are same as that of ROWs
    
# ==== (3) CONSTRAINTS (1) & (5) are found to be conflicting and only (1) is considered 
#           in this code 
 

# In[1.0]: Problem statement 

# Rows need to be divided into N groups such that members of the group have the same \
#     value of D. N can be any number between 1 to the total number of rows. \
#     N can be known or unknown. This means that a column can at max have N different \
#     values. This N can be the same across columns or there can be a max Nmax such that\ 
#     each column’s N is lower than or equal to Nmax.

# If N is known, it needs to be taken as a constraint. If not, the algorithm \
# should identify the optimal value of N satisfying other constraints and objective

# We have a table which has for any P x T x D combination where P is in P1…Pn and \
#     T is in T1..Tn, and D is any admissible value for the decision variable \
#     in {10,20,30,40,50,60,70} lookup values for three quantities U, R and M. \
#     U values are all positive integers. \
#     R and M values are mostly positive integers but can be positive decimal numbers as well

p_var = list(range(0,10,1))
t_var = list(range(1,10,1))
d_var = list(range(10,80,10))

base_df_p = pd.DataFrame(p_var).rename(columns={0:'p_val'})
base_df_t = pd.DataFrame(t_var).rename(columns={0:'t_val'})
base_df_d = pd.DataFrame(d_var).rename(columns={0:'d_val'})

base_df_p.loc[:,'col_com'] =  np.random.randint(0,3,len(base_df_p))
base_df_t.loc[:,'col_com'] =  np.random.randint(0,5,len(base_df_t))
base_df_d.loc[:,'col_com'] =  np.random.randint(0,5,len(base_df_d))

df_groups = cp.deepcopy(base_df_d[["d_val"]])

# In[1.0]: creating matrxi P X T X D

# creating a data frame to mark the group indices for D values 
temp_df = pd.DataFrame()
for i in set(df_groups.d_val):
    df_for_looping = pd.DataFrame(np.arange(0,np.random.randint(1,4),1)).drop_duplicates().rename(columns={0:'group_index'})
    df_for_looping.loc[:,'d_val'] = np.repeat(i, len(df_for_looping))
    
    temp_df = pd.concat([temp_df, df_for_looping])
    
    del df_for_looping
    
temp_df_group_count = temp_df.groupby('d_val').count().reset_index().rename(columns=
                                                                            {'group_index':'group_count'})    
final_df_group_count = temp_df_group_count.merge(temp_df, on='d_val')
final_df_group_count.loc[:,'group_index'] = final_df_group_count['group_index']+1
    

# In[1.0]: creating matrxi P X T X D
df_p_t_d = base_df_p.merge(base_df_t, how='left'
                           ).merge(base_df_d, how='left')

df_p_t_d = df_p_t_d.dropna().reset_index(drop=True)
df_p_t_d.loc[:,'t_val'] = df_p_t_d['t_val'].apply(int)
df_p_t_d.loc[:,'d_val'] = df_p_t_d['d_val'].apply(int)

df_p_t_d.loc[:,'M_val'] = np.random.randint(0,10,len(df_p_t_d))

df_p_t_d = df_p_t_d.drop(columns = ['col_com'], axis=0)


 
# In[1.0]: variable creation and adding into the table 
df_p_t_d_var = df_p_t_d.merge(final_df_group_count, on="d_val")

df_p_t_d_var['x_var'] = df_p_t_d_var['p_val'].apply(str)+'p_'+\
                        df_p_t_d_var['t_val'].apply(str)+'t_'+\
                        df_p_t_d_var['d_val'].apply(str)+'d_'+\
                        df_p_t_d_var['group_index'].apply(str)+'g'
                                

# In[1.0]: variable creation and adding into the table 

# U values are all positive integers. 
# R and M values are mostly positive integers but can be positive decimal numbers as well.

df_p_t_d_var.loc[:,'M_val'] = df_p_t_d_var['M_val']+np.random.randint(10,100,1)
df_p_t_d_var.loc[:,'U_val'] = np.random.randint(10,100,len(df_p_t_d_var))
df_p_t_d_var.loc[:,'R_val'] = np.random.randint(10,100,len(df_p_t_d_var))


# In[1.0]: defining variable matirx                             
var_df = cp.deepcopy(df_p_t_d_var)
var_df = var_df[["x_var"]]

# defining 
prob = pulp.LpProblem("MAXIMIZATION",LpMaximize)
x = pulp.LpVariable.dicts('x', df_p_t_d_var.x_var, cat=LpContinuous)
y = pulp.LpVariable.dicts('y', df_p_t_d_var.x_var, cat=LpBinary) 

Big_M = 100

# In[1.0]: defining objective 
for i in set(df_p_t_d_var.d_val):
    prob += lpSum(x[var]* coeff for var,coeff in zip(df_p_t_d_var[df_p_t_d_var.d_val==i].x_var, 
                                                              df_p_t_d_var[df_p_t_d_var.d_val==i].M_val)), 'objective'


# In[1.0]: defining objective 
for i in set(df_p_t_d_var.x_var):
    prob += x[i] >= 25*(1-y[i]), 'big_m_con@'+str(i)

    
# In[1.0]: constraints set 1: On number of groups 
    # all the variables for a given D_val should be >= 0 and <= group_count 
#  If N is known, it needs to be taken as a constraint. If not, the algorithm 
# should identify the optimal value of N satisfying other constraints and objective
    
for i in set(df_p_t_d_var.d_val):
    temp_df_2 = df_p_t_d_var[df_p_t_d_var.d_val==i].reset_index(drop=True)
    for (k,l) in set(zip(temp_df_2.p_val, temp_df_2.t_val)):
        prob += lpSum(x[var]  for var in temp_df_2[(temp_df_2.p_val==k)&(temp_df_2.t_val==l)].x_var) <= 2 , 'con_grouping_low_'+str(i)+'_'+str(k)+'_'+str(l)
        prob += lpSum(x[var]  for var in temp_df_2[(temp_df_2.p_val==k)&(temp_df_2.t_val==l)].x_var) >= 0 , 'con_grouping_up_'+str(i)+'_'+str(k)+'_'+str(l)

                                        
# In[Constraint: 3]: 
# The total number of unique values of the decision variable appearing in each 
#                                           row can be at least I and at most J
        

        
# In[1.0]: The total number of unique values of the decision variable 
# appearing in each row can be at least I and at most J
        # HERE I is 0 and J is 4

I = 2 
J = 10
print ('The total number of unique values of the decision variable appearing \
               in each row can be at least...' + str(I) + 'and at most ...' +str( J))               
for i in set(df_p_t_d_var.d_val):
    prob += lpSum(x[var]  for var in df_p_t_d_var[(df_p_t_d_var.d_val==i)].x_var) >= I , 'unique_dec_var_low_'+str(i)
    prob += lpSum(x[var]  for var in df_p_t_d_var[(df_p_t_d_var.d_val==i)].x_var) <= J , 'unique_dec_var_high_'+str(i)


# In[1.0]: Duration of each level for a row is at least L and at most M, \
#              duration means successive cells having the same value of decision variable

# Here it's been assumed that 'duration' means time duration also 'CELLS' means 'Groups'   
# Duration is SUM (var*time)  

L = 0 
M = 10
for i in set(df_p_t_d_var.d_val):
        prob += lpSum(x[var]*time  for var, time in zip(df_p_t_d_var[(df_p_t_d_var.d_val==i)].x_var,
                                                        df_p_t_d_var[(df_p_t_d_var.d_val==i)].t_val)) >= L, 'duration_con_low'+str(i)
        
        prob += lpSum(x[var]*time  for var, time in zip(df_p_t_d_var[(df_p_t_d_var.d_val==i)].x_var,
                                                        df_p_t_d_var[(df_p_t_d_var.d_val==i)].t_val)) <= M, 'duration_con_high'+str(i)
 

# In[Rule:6]: In a row, differences between successive D values are all at least X and at most Y

print ('Since the followig constraint is a positional constraint and may not be driven by \
       the choice of variables it has been omitted ...')
       
print ('In a row, differences between successive D values are all at least \
       X and at most Y')


# In[Rule:7]: For each row, the sum of U cannot exceed a Umax that is provided as external input

# Here a row corresponds to sepcific D_val and hence all the variables comes under
       # D were summed to U 
       
u_max = 1000 
for i in set(df_p_t_d_var.d_val):
    prob += lpSum(x[var]*Uval  for var, Uval in zip(df_p_t_d_var[(df_p_t_d_var.d_val==i)].x_var,
                                                    df_p_t_d_var[(df_p_t_d_var.d_val==i)].U_val)) <= u_max, 'upper_limit_on_umax'+str(i)


# In[Rule.8]: The value differences between the unique N levels in any column \
    #  is at least equal to A and at most B




# In[]
# 8. The total sum of U in the grid is at least Uthresold, there might be \
#     Uthresholds for each row. 
# 9. The total sum of M in the grid is at least Mthresold. 
# 10. The decision variables in a row might need to follow ascending or \
#     descending order as per an input. (If not either of the case, this \
#                                        constraint will not be applied) 
# 11. Each cell should get a maximum of one value of the decision variable.







    
    
    
    
    
    
    


