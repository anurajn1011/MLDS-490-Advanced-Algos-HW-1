# Notes for Question 1

## Shared Assumptions
These are assumptions which all three parts are based on. 

* State = number waiting from 0 to 200; action 1 = dispatch, 0 = wait.
* Order within a period: observe, decide, dispatch (remove up to K), pay holding cost on those left, then arrivals join. Arrivals can’t board the current shuttle.
* Cost = cf·(dispatch) + ch·(post-dispatch queue AKA how many left at station). It is a cost, so we minimize.
* Arrivals ~ Unif{1..5}, so each has probability 0.2.
* Overflow beyond 200 is lost with no penalty (probability pools on state 200).
* At most one shuttle per period.
* Ties go to no dispatch (strict <).

## Enumeration

* For this first part of the question, I assume that $V_T = 0$ where $T = 500$. As such, the first terminal value is at $T-1$.
* The policy is time-dependent. Only policy\[0\] is the relevant answer, and the last few periods are an end-of-horizon artifact.

## Value Iteration

* Same shared assumptions listed above. We start from a Value function of all 0's. We stop iterating when the largest difference between the value functions, at some coordinate, falls beneath a threshold of 1e-8. 
* Then, we choose the policy via greedy search. 

## Policy Iteration

* We initalize our policy as one where we never dispatch.
* Policy's are evaluated within a threshold as well for the value function.
* We cease when the polcy has not changed between iterations. 