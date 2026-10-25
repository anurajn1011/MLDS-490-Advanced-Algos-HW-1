# Enumeration

For this first part of the question, I assume that $V_T = 0$ where $T = 500$. As such, the first terminal value is at $T-1$.

The action space is binary, we either dispatch (1) or don't (0). Additionally, since adding more individuals to the station overflows capacity, we remove the extra individuals with no penalty/cost incurred for losing the customer. 

# Value Iteration

Same assumptions as before. 