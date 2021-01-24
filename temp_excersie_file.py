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
print ('Please read the assumptions before executing and analysing \n \
               solution of this file ...\n') 

# ====   The following assumptions are made and the code will strictcly 
#        restricted to the same. 

# ==== (1) ROWS, LEVELS and GROUPS are considered to representing the same information 
# ==== (2) GRIDS are collection of ROWS 
    
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
base_df_t.loc[:,'col_com'] =  np.random.randint(1,3,len(base_df_t))
base_df_d.loc[:,'col_com'] =  np.random.randint(0,3,len(base_df_d))

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

u = pulp.LpVariable.dicts('u', df_p_t_d_var.x_var, cat=LpInteger)
m = pulp.LpVariable.dicts('m', df_p_t_d_var.x_var, lowBound = 0, cat=LpContinuous)
r = pulp.LpVariable.dicts('r', df_p_t_d_var.x_var, lowBound = 0, cat=LpContinuous)
y = pulp.LpVariable.dicts('y', df_p_t_d_var.x_var, lowBound = 0, cat=LpBinary) 

Big_M = 25

# In[Objective]:
for i in set(df_p_t_d_var.d_val):
    prob += lpSum(u[var]  for var in df_p_t_d_var[df_p_t_d_var.d_val==i].x_var), 'objective_to_maximize_M'

 
# In[Constraint-3]: 
    # The total number of unique values of the decision variable \
    #  appearing in each row can be at least I and at most J

    
for i in set(df_p_t_d_var.x_var):
    prob += u[i] >= Big_M*(1-y[i]), 'big_m_con_on_u@'+str(i)
    prob += m[i] >= Big_M*(1-y[i]), 'big_m_con_on_m@'+str(i)
    prob += r[i] >= Big_M*(1-y[i]), 'big_m_con_on_r@'+str(i)

for i in set(df_p_t_d_var.p_val):
    prob += lpSum(y[var] for var in (df_p_t_d_var[df_p_t_d_var.p_val==i].x_var)) >= 10 , 'low_limit_on_row_var_'+str(i)
    prob += lpSum(y[var] for var in (df_p_t_d_var[df_p_t_d_var.p_val==i].x_var)) <= 5000 , 'upper_limit_on_row_var_'+str(i)


# In[Constraint-4]: 
 # Duration of each level for a row is at least L and at most M, 
    # duration means successive cells having the same value of decision variable

con_4_low_limit = 10 
con_4_up_limit = 5000 
    
for i in set(df_p_t_d_var.p_val):
    temp_df_2 = df_p_t_d_var[df_p_t_d_var.p_val==i].reset_index(drop=True)
    for j in set(temp_df_2.d_val):
        # print ([i,j])
        prob += lpSum(y[var]*t_val  for var,t_val in zip(temp_df_2[temp_df_2.d_val==j].x_var, 
                                                         temp_df_2[temp_df_2.d_val==j].t_val)) >= con_4_low_limit, 'low_limit_on_time'+str(i)+'_'+str(j) 
                  
        prob += lpSum(y[var]*t_val  for var,t_val in zip(temp_df_2[temp_df_2.d_val==j].x_var, 
                                                         temp_df_2[temp_df_2.d_val==j].t_val)) <= con_4_up_limit, 'upper_limit_on_time'+str(i)+'_'+str(j) 

              
# In[Constraint-5]: 
 # In a row, differences between successive D values are all at least X and at most Y 

print ('In[Constraint-5]:')
print ('this has been omitted since X, Y are user defined and can be made \n \
               redundent using X =0 and Y can be 60 (70-10) \n')



# In[Constraint-6]: 
    # For each row, the sum of U cannot exceed a Umax that is provided as external input               

print('In[Constraint-6]:')
print('For each row, the sum of U cannot exceed a Umax that is provided as external input \n')

for i in set(df_p_t_d_var.p_val):
    prob += lpSum(u[var]  for var in df_p_t_d_var[df_p_t_d_var.p_val==i].x_var) <= 999999, 'upper_bound_on_U'+str(i)
                  


# In[Constraint-7]: 
# The value differences between the unique N levels in any column is \
    # at least equal to A and at most B
# In a column (for all Ti), the difference between any two D's present in \
    # it must be at least A and at most B.  

print ('In[Constraint-7]:')
print ('this has been omitted since A, B are user defined and can be made \n \
               redundent using A =0 and B can be 60 (70-10) \n')


# In[Constraint-8]: 
    # The total sum of U in the grid is at least Uthresold, there might be \
    # Uthresholds for each row.

print ('In constriat -8 ...\n \
       The total sum of U in the grid is at least Uthresold, there might be \n \
       Uthresholds for each row. \n ')

u_thresh = []               
for i in set(base_df_p.p_val):
    # print (i)
    u_thresh.append([i, np.random.randint(10,100,1)[0]])
u_thresh_df = pd.DataFrame(u_thresh).rename(columns={0:'p_val',1:'u_threshold'})

for i in set(df_p_t_d_var.p_val):
    prob += lpSum(u[var]  for var in df_p_t_d_var[df_p_t_d_var.p_val==i].x_var) <= u_thresh_df[u_thresh_df.p_val==0]['u_threshold'].item(), 'U_threshold_row_'+str(i)
 
               
# In[Constraint-9]: 
    # The total sum of M in the grid is at least Mthresold.

print ('In constriat -9 ...The total sum of M in the grid is at least Mthresold. \n ')

m_thresh = []               
for i in set(base_df_p.p_val):
    # print (i)
    m_thresh.append([i, np.random.randint(10,100,1)[0]])
m_thresh_df = pd.DataFrame(m_thresh).rename(columns={0:'p_val',1:'m_threshold'})

for i in set(df_p_t_d_var.p_val):
    prob += lpSum(m[var]  for var in df_p_t_d_var[df_p_t_d_var.p_val==i].x_var) >= m_thresh_df[m_thresh_df.p_val==0]['m_threshold'].item(), 'M_threshold_row_'+str(i)
 

# In[Constraint-10]:                
    #  The decision variables in a row might need to follow ascending or \
    #   descending order as per an input. (If not either of the case, this \
    #   constraint will not be applied)

print ('In constriat -10 ...\n ')
print ('since the preset model has not accounted for sequecing of the variables \n \
       this constraint has been omitted \n ')


# In[Constraint-11]:  Each cell should get a maximum of one value of the decision variable.

print ('In constriat -11 ...\n ')
print ('Here a cell is assumed to be the one associated with individual variable \n \
           present in a decision variable (D) ... \n ')
           
for i in set(df_p_t_d_var.p_val):
    temp_df_2 = df_p_t_d_var[df_p_t_d_var.p_val==i].reset_index(drop=True)
    for j in set(temp_df_2.d_val):
        # print ([i,j])
        prob += lpSum([u[var]  for var in temp_df_2[temp_df_2.d_val==j].x_var]) >= 0, 'max_decision_var'+str(i)+'_'+str(j) 
               

# In[Model Solving Block]:

time_to_start = time.time()

prob.solve()

time_to_end = time.time()

solve_time = time_to_end - time_to_start      

print ('solver run time ...: \n', solve_time)

print ('\n')

print ('writing the model into lp format... \n', )

prob.writeLP('model_lp_file.txt')

print ('objective', value(prob.objective), 'and status' ,prob.status, '\n')

# In[Solution Extraction]:
groups_list = []
for var in set(df_p_t_d_var.x_var):
    if y[var].value()>=1: 
        if (u[var].value()>=0 and  m[var].value()>=0 and r[var].value()>=0):
            if (u[var].value()>0 or m[var].value()>0):
                groups_list.append([var, u[var].value(), m[var].value(), r[var].value(), y[var].value()])
                
final_sol_df = pd.DataFrame(groups_list).rename(columns={0:'var_name', 
                                                         1:'U_val', 
                                                         2:'M_val', 
                                                         3:'R_val',
                                                         4:'frequency'}) 

# apply(lambda x : x[0]+x[1] if x[2]==2 else x[0] , axis =1)

final_df_with_p_index = pd.concat([final_sol_df.var_name.apply(lambda x: pd.Series(str(x).split("_")[0][0])).rename(columns={0:'p_index'}), 
                                   final_sol_df], axis=1)

final_df_with_t_index = pd.concat([final_sol_df.var_name.apply(lambda x: pd.Series(str(x).split("_")[1][0])).rename(columns={0:'t_index'}), 
                                   final_df_with_p_index], axis=1)

final_df_with_d_index = pd.concat([final_sol_df.var_name.apply(lambda x: pd.Series(str(x).split("_")[2][0:2])).rename(columns={0:'d_index'}), 
                                   final_df_with_t_index], axis=1)

final_df_with_g_index = pd.concat([final_sol_df.var_name.apply(lambda x: pd.Series(str(x).split("_")[3][0])).rename(columns={0:'g_index'}), 
                                   final_df_with_d_index], axis=1)

    


