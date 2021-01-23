# impact_analytics_task

-- The objective of this excercise is to develop a mathematical model to identify the values for U, M, R columns across a ROW while meeting specific requirements.

A group consists of combination of rows comprising various values for  M, U and R for a give D value, and with various predefined T values (time durations).   
A set of such groups is called as ROW. 
A set of such ROWs is called as GRID. 

Since the values of M, U, R are to be identified, three different variables to be created and among these only variables corresponding to M need to be considred in the objective as a Maximization objective. 

The following assumptions are made in modelling the model. 

1. A priori chocies were made such that there are pre-defined 'possible' number of groups for each D value. However, solver will find the right number of groups only after finding the choice. 

2.    
