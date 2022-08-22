#Input starting values of p, q, r
p_initial=1/3
q_initial=1/3
r_initial=1/3

#Input observed data
NA=725
NB=258
NAB=72
NO=1073
N=2128

#Expected value of missing data
NAA_exp=(p_initial**2*NA)/(p_initial**2+2*p_initial*q_initial)
NBB_exp=(q_initial**2*NB)/(q_initial**2+2*q_initial*r_initial)
NAO_exp=(2*p_initial*r_initial*NA)/(p_initial**2+2*p_initial*r_initial)
NBO_exp=(2*q_initial*r_initial*NB)/(q_initial**2+2*q_initial*r_initial)

#Maximum likelihood after applying EM
p_new=(2*NAA_exp+NAO_exp+NAB)/(2*N)
q_new=(2*NBB_exp+NBO_exp+NAB)/(2*N)
r_new=(2*NO+NAO_exp+NBO_exp)/(2*N)

print(NAA_exp,NAO_exp,NBB_exp,NBO_exp)
print(p_new, q_new, r_new)