These are programs for my paper "Pricing model performance and the
two-pass cross-sectional regression methodology" (with Cesare Robotti 
and Jay Shanken).  If you have questions, comments, or bug reports, 
please send them to kan@chass.utoronto.ca

Version 1.0: 5/29/2009 initial release
Version 1.1: 8/17/2010 fix a bug in nonnested.m, for the test of
             rho_1^2=rho_2^2, previous version does not divide xi by Q0.
Version 1.2: 11/9/2011 fix a bug in nw.m, for the automatic lag length
             selection part, the definition of s0 and s1 were switched.

Raymond Kan
Rotman School of Management
University of Toronto


csrw.m: perform cross-sectional regression with fixed weighting matrix
csrgls.m: perform GLS cross-sectional regression
csrwls.m: perform WLS cross-sectional regression
nw.m: Newey-West estimate of covariance matrix with
      possible automatic lag selection
nested.m: Nested models comparison
nonnested.m: Nonnoested models comparison
linchi2.m: Program to compute the cdf of a weighted chi-squared distribution
