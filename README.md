# Guide Lines to understand the problem statement 

Problem:
We define a matrix as below, P1…Pn come from a class of entities, columns are Time periods and cells contain the decision variables. In a given instance of the problem, P1…Pn is variable, and T1...Tn can also be variable but there is a max Tn. P1...Pn and T1...Tn are inputs to the problem and are given. The problem needs to determine the best set of values for the cells, from a known set {10,20,30,40,50,60,70}


We have a table which has for any P x T x D combination where P is in P1…Pn and T is in T1..Tn, and D is any admissible value for the decision variable in {10,20,30,40,50,60,70} lookup values for three quantities U, R and M. U values are all positive integers. R and M values are mostly positive integers but can be positive decimal numbers as well.

# The objective is to maximize the sum of M across the entire grid. 

-- The row-level constraints are as follows: 

1. Rows need to be divided into N groups such that members of the group have the same value of D. N can be any number between 1 to the total number of rows. N can be known or unknown. This means that a column can at max have N different values. This N can be the same across columns or there can be a max Nmax such that each column’s N is lower than or equal to Nmax. 

2. If N is known, it needs to be taken as a constraint. If not, the algorithm should identify the optimal value of N satisfying other constraints and objective 

3. The total number of unique values of the decision variable appearing in each row can be at least I and at most J 

4. Duration of each level for a row is at least L and at most M, duration means successive cells having the same value of decision variable 

5. In a row, differences between successive D values are all at least X and at most Y 

6. For each row, the sum of U cannot exceed a Umax that is provided as external input

The column constraints can be: 

7. The value differences between the unique N levels in any column is at least equal to A and at most B 

-- The overall constraints are as follows: 

8. The total sum of U in the grid is at least Uthresold, there might be Uthresholds for each row. 
9. The total sum of M in the grid is at least Mthresold. 
10. The decision variables in a row might need to follow ascending or descending order as per an input. (If not either of the case, this constraint will not be applied) 
11. Each cell should get a maximum of one value of the decision variable.
