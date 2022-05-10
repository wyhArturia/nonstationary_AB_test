import numpy as np
from scipy.stats import poisson
import multiprocessing

def fit_mean(t): # t\in [0,7]
    if (t<=0.3):
        f = 0.65-(0.35)/0.3*t
    if (0.3<t<=0.6):
        f = 0.3-(0.11)/0.3*(t-0.3)
    if (0.6<t<=1):
        f = 0.19+0.31/(0.4)*(t-0.6)
    if (1<t<=1.2):
        f = 0.5-0.15/0.2*(t-1)
    if (1.2<t<=1.7):
        f = 0.35-0.19/0.5*(t-1.2)
    if (1.7<t<=2):
        f = 0.16+0.44/0.3*(t-1.7)
    if (2<t<=2.4):
        f = 0.6-0.3/0.4*(t-2)
    if (2.4<t<=2.7):
        f = 0.3-0.11/0.3*(t-2.4)
    if (2.7<t<=3):
        f = 0.19+0.13/0.3*(t-2.7)
    if (3<t<=3.2):
        f = 0.32+0.05/0.2*(t-3)
    if (3.2<t<=3.5):
        f = 0.37-0.16/0.3*(t-3.2)
    if (3.5<t<=3.7):
        f = 0.21-0.04/0.2*(t-3.5)
    if (3.7<t<=4):
        f = 0.17+0.2/0.3*(t-3.7)
    if (4<t<=4.1):
        f = 0.37+0.13/0.1*(t-4)
    if (4.1<t<=4.7):
        f = 0.5-0.32/0.6*(t-4.1)
    if (4.7<t<=5):
        f = 0.18+0.16/0.3*(t-4.7)
    if (5<t<=5.2):
        f = 0.34+0.16/0.3*(t-5)
    if (5.2<t<=5.5):
        f = 0.5-0.25/0.3*(t-5.2)
    if (5.5<t<=5.8):
        f = 0.25-0.14/0.3*(t-5.5)
    if (5.8<t<=6.2):
        f = 0.11+0.43/0.4*(t-5.8)
    if (6.2<t<=6.4):
        f = 0.54-0.24/0.2*(t-6.2)
    if (6.4<t<=6.8):
        f = 0.3-0.17/0.4*(t-6.4)
    if (6.8<t<=7):
        f = 0.13+0.04/0.2*(t-6.8)
    f = f*7
    return f

def fit_sd(t): #t\in [0,7]
    if (t<=0.3):
        f = 0.8-0.4/0.3*(t)
    if (0.3<t<=0.6):
        f = 0.4-0.04/0.3*(t-0.3)
    if (0.6<t<=1):
        f = 0.36+0.28/0.4*(t-0.6)
    if (1<t<=1.2):
        f = 0.64-0.14/0.2*(t-1)
    if (1.2<t<=1.7):
        f = 0.5-0.17/0.5*(t-1.2)
    if (1.7<t<=2):
        f = 0.33+0.31/0.3*(t-1.7)
    if (2<t<=2.4):
        f = 0.64-0.24/0.4*(t-2)
    if (2.4<t<=2.7):
        f = 0.4-0.05/0.3*(t-2.4)
    if (2.7<t<=3):
        f = 0.35+0.15/0.3*(t-2.7)
    if (3<t<=3.2):
        f = 0.5+0.05/0.2*(t-3)
    if (3.2<t<=3.5):
        f = 0.55-0.18/0.3*(t-3.2)
    if (3.5<t<=3.7):
        f = 0.37-0.03/0.2*(t-3.5)
    if (3.7<t<=4):
        f = 0.34+0.21/0.3*(t-3.7)
    if (4<t<=4.1):
        f = 0.55+0.1/0.1*(t-4)
    if (4.1<t<=4.7):
        f = 0.65-0.29/0.6*(t-4.1)
    if (4.7<t<=5.2):
        f = 0.36+0.24/0.5*(t-4.7)
    if (5.2<t<=5.8):
        f = 0.6-0.3/0.6*(t-5.2)
    if (5.8<t<=6.2):
        f = 0.3+0.35/0.4*(t-5.8)
    if (6.2<t<=6.4):
        f = 0.65-0.27/0.2*(t-6.2)
    if (6.4<t<=6.7):
        f = 0.38-0.06/0.3*(t-6.4)
    if (6.7<t<=7):
        f = 0.32+0.1/0.3*(t-6.7)
    return f

def fit_mean_treat(t):
    if (t<=0.5):
        tmp = 0.3+0.17/0.5*(t)
    if (0.5<t<=1):
        tmp = 0.47-0.27/0.5*(t-0.5)
    if (1<t<=1.5):
        tmp = 0.2+0.25/0.5*(t-1)
    if (1.5<t<=2):
        tmp = 0.45-0.25/0.5*(t-1.5)
    if (2<t<=2.5):
        tmp = 0.2+0.46*(t-2)
    if (2.5<t<=3):
        tmp = 0.43-0.36*(t-2.5)
    if (3<t<=3.5):
        tmp = 0.25+0.46*(t-3)
    if (3.5<t<=4):
        tmp = 0.48-0.4*(t-3.5)
    if (4<t<=4.5):
        tmp = 0.28+0.46*(t-4)
    if (4.5<t<=5):
        tmp = 0.51-0.48*(t-4.5)
    if (5<t<=5.5):
        tmp = 0.27+0.44*(t-5)
    if (5.5<t<=6):
        tmp = 0.71-0.62*(t-5.5)
    if (6<t<=6.5):
        tmp = 0.4+0.84*(t-6)
    if (6.5<t<=7):
        tmp = 0.82-0.76*(t-6.5)
    return (2*tmp+fit_mean(t))

def fit_sd_treat(t):
    if (int(t-0.5)<int(t)):
        s = t-int(t)
        f = 0.03+0.04/0.5*s
    else:
        s = t-int(t)
        f = 0.07-0.04/0.5*(s-0.5)
    return f+fit_sd(t)

def fit_arrival_rate(t):
    if (t<=0.5):
        f = 0.3+0.4/0.5*(t)
    if (0.5<t<=1):
        f = 0.7-0.35/0.5*(t-0.5)
    if (1<t<=1.5):
        f = 0.35+0.3/0.5*(t-1)
    if (1.5<t<=2):
        f = 0.65-0.3/0.5*(t-1.5)
    if (2<t<=2.5):
        f = 0.35+0.25/0.5*(t-2)
    if (2.5<t<=3):
        f = 0.6-0.25/0.5*(t-2.5)
    if (3<t<=3.5):
        f = 0.35+0.2/0.5*(t-3)
    if (3.5<t<=4):
        f = 0.55-0.2/0.5*(t-3.5)
    if (4<t<=4.5):
        f = 0.35+0.1/0.5*(t-4)
    if (4.5<t<=5):
        f = 0.45-0.15/0.5*(t-4.5)
    if (5<t<=5.5):
        f = 0.3+0.4/0.5*(t-5)
    if (5.5<t<=6):
        f = 0.7-0.35/0.5*(t-5.5)
    if (6<t<=6.5):
        f = 0.35+0.55/0.5*(t-6)
    if (6.5<t<=7):
        f = 0.9-0.4/0.5*(t-6.5)
    return f

def est_true_effect(t):
    return (fit_mean_treat(t)-fit_mean(t))*fit_arrival_rate(t)

from scipy import integrate

res_1 = 0
res_2 = 0

for i in range(14):
    t_1 = i*0.5
    t_2 = (i+1)*0.5
    fArea1,err1 = integrate.quad(fit_arrival_rate,t_1,t_2)
    fArea2,err2 = integrate.quad(est_true_effect,t_1,t_2)
    res_1 += fArea1
    res_2 += fArea2

norm_num = res_1/7
true_effect = res_2/res_1

prob_area = np.zeros((6,300))
k_value = np.array([14,21,42,84,168])
prob_area[0,0] = 1
for j in range(4):
    K = k_value[j]
    for i in range(K):
        t_1 = (i)*7/K
        t_2 = (i+1)*7/K
        area_tmp,err_tmp = integrate.quad(fit_arrival_rate,t_1,t_2)
        prob_area[j+1,i] = area_tmp*2/7

def generate_arrival_process(lambda_1,T):
    total_number = poisson.rvs(lambda_1*2)
    arrival_time = []
    n = 0
    for i in range(total_number):
        tmp = np.random.random()
        tmp_2 = np.random.random()
        if tmp_2<fit_arrival_rate(tmp*7):
            arrival_time.append(tmp)
            n += 1
    arrival_time_out = T*np.sort(np.array(arrival_time))
    return n, arrival_time_out

def generate_arrival_process_sec_4(n,T):
    arrival_time = []
    for i in range(n):
        tmp = np.random.random()
        arrival_time.append(tmp)
    arrival_time_out = T*np.sort(np.array(arrival_time))
    return arrival_time_out

def generate_outcome(t,treat_label):
    tmp = np.random.normal()
    if treat_label == 1:
        res = fit_mean_treat(t)+fit_sd_treat(t)*tmp
    else:
        res = fit_mean(t)+fit_sd(t)*tmp
    return res

def changing_p(t):
    tmp = int(t*2)
    p = 0.1+tmp*0.4/13
    return p


def nonstationary_abtest(lambda_bar, T):
    effect_K = np.zeros((6))
    effect_1 = np.zeros((6))
    effect_Q = np.zeros((6))
    effect_naive = np.zeros((1))

    kn = int(lambda_bar ** 0.4 * 3 / 7) * 7
    for i in range(kn):
        t_1 = (i) * 7 / kn
        t_2 = (i + 1) * 7 / kn
        area_tmp, err_tmp = integrate.quad(fit_arrival_rate, t_1, t_2)
        prob_area[5, i] = area_tmp * 2 / 7

    n, arrival_time = generate_arrival_process(lambda_bar, T)

    outcome = np.zeros((n))

    # experiment design: i.i.d. bernoulli data with p=0.5
    z = np.zeros((n))
    for i in range(n):
        tmp = np.random.random()
        if tmp > 0.5:
            z[i] = 1
            outcome[i] = generate_outcome(arrival_time[i], 1)
        else:
            z[i] = 0
            outcome[i] = generate_outcome(arrival_time[i], 0)

    for l in range(6):
        array_tmp = np.array([1, 7, 21, 42, 84, kn])
        K = array_tmp[l]
        num_outcome = np.zeros((K, 2))
        sum_outcome = np.zeros((K, 2))
        est_treat_effect = 0
        for i in range(n):
            tmp = int(arrival_time[i] / 7 * K)
            if z[i] == 1:
                num_outcome[tmp, 1] += 1
                sum_outcome[tmp, 1] += outcome[i]
            else:
                num_outcome[tmp, 0] += 1
                sum_outcome[tmp, 0] += outcome[i]
        for i in range(K):
            if ((num_outcome[i, 1] != 0) and (num_outcome[i, 0] != 0)):
                est_treat_effect += (num_outcome[i, 1] + num_outcome[i, 0]) * (
                            sum_outcome[i, 1] / num_outcome[i, 1] - sum_outcome[i, 0] / num_outcome[i, 0])
        est_treat_effect = est_treat_effect / n
        effect_K[l] = est_treat_effect

    # time-group randomization
    z = np.zeros((n))
    n_0 = n
    pair = int(n / 2)
    n = 2 * pair
    for i in range(pair):
        tmp = np.random.random()
        if (tmp < 0.5):
            z[2 * i] = 1
            outcome[2 * i] = generate_outcome(arrival_time[2 * i], 1)
            z[2 * i + 1] = 0
            outcome[2 * i + 1] = generate_outcome(arrival_time[2 * i + 1], 0)
        else:
            z[2 * i] = 0
            outcome[2 * i] = generate_outcome(arrival_time[2 * i], 0)
            z[2 * i + 1] = 1
            outcome[2 * i + 1] = generate_outcome(arrival_time[2 * i + 1], 1)
    for l in range(6):
        array_tmp = np.array([1, 7, 21, 42, 84, kn])
        K = array_tmp[l]
        num_outcome = np.zeros((K, 2))
        sum_outcome = np.zeros((K, 2))
        est_treat_effect = 0
        for i in range(n):
            tmp = int(arrival_time[i] / 7 * K)
            if z[i] == 1:
                num_outcome[tmp, 1] += 1
                sum_outcome[tmp, 1] += outcome[i]
            else:
                num_outcome[tmp, 0] += 1
                sum_outcome[tmp, 0] += outcome[i]
        for i in range(K):
            if ((num_outcome[i, 1] != 0) and (num_outcome[i, 0] != 0)):
                est_treat_effect += (num_outcome[i, 1] + num_outcome[i, 0]) * (
                            sum_outcome[i, 1] / num_outcome[i, 1] - sum_outcome[i, 0] / num_outcome[i, 0])
        est_treat_effect = est_treat_effect / n
        effect_1[l] = est_treat_effect

    # The case that p(t) changes, see Section 4
    n = n_0
    arrival_time = generate_arrival_process_sec_4(n, T)
    z = np.zeros((n))
    for i in range(n):
        tmp = np.random.random()
        if tmp < changing_p(arrival_time[i]):
            z[i] = 1
            outcome[i] = generate_outcome(arrival_time[i], 1)
        else:
            z[i] = 0
            outcome[i] = generate_outcome(arrival_time[i], 0)

    for l in range(6):
        array_tmp = np.array([1, 14, 21, 42, 84, kn])
        K = array_tmp[l]
        num_outcome = np.zeros((K, 2))
        sum_outcome = np.zeros((K, 2))
        est_treat_effect = 0
        for i in range(n):
            tmp = int(arrival_time[i] / 7 * K)
            if z[i] == 1:
                num_outcome[tmp, 1] += 1
                sum_outcome[tmp, 1] += outcome[i]
            else:
                num_outcome[tmp, 0] += 1
                sum_outcome[tmp, 0] += outcome[i]
        if K == 1:
            est_naive = (sum_outcome[0, 1] / num_outcome[0, 1] - sum_outcome[0, 0] / num_outcome[0, 0])
        for i in range(K):
            if ((num_outcome[i, 1] != 0) and (num_outcome[i, 0] != 0)):
                est_treat_effect += prob_area[l, i] * (
                            sum_outcome[i, 1] / num_outcome[i, 1] - sum_outcome[i, 0] / num_outcome[i, 0])
        effect_Q[l] = est_treat_effect
        effect_naive[0] = est_naive

    res = np.zeros((19))
    for i in range(6):
        res[i] = effect_1[i]
        res[i + 6] = effect_K[i]
        res[i + 12] = effect_Q[i]
    res[18] = effect_naive
    return res


def main_exp(ii):
    tmp = nonstationary_abtest(16000,7)
    return tmp


if __name__ == "__main__":
    tot = 100000
    # effect = np.zeros((tot))
    # asymp_sigma_est = np.zeros((tot))

    pool_obj = multiprocessing.Pool()
    res_out = pool_obj.map(main_exp, range(0, tot))
    res_out = np.array(res_out)

    effect_1 = np.zeros((tot, 6))
    effect_K = np.zeros((tot, 6))
    effect_Q = np.zeros((tot, 6))
    mse_1 = np.zeros((6))
    mse_2 = np.zeros((6))
    mse_3 = np.zeros((6))
    est_var = np.zeros((6))
    est_biased = np.zeros((6))
    effect_naive = np.zeros((tot))
    for j in range(tot):
        for l in range(6):
            effect_1[j, l] = res_out[j, l]
            effect_K[j, l] = res_out[j, l + 6]
            effect_Q[j, l] = res_out[j, l + 12]
        effect_naive[j] = res_out[j, 18]

    for i in range(6):
        res = effect_1[:, i]
        mse_1[i] = np.sum((res - true_effect) ** 2) / tot

    for i in range(6):
        res = effect_K[:, i]
        mse_2[i] = np.sum((res - true_effect) ** 2) / tot

    for i in range(6):
        res = effect_Q[:, i]
        est_biased[i] = np.sum(res) / tot
        mse_3[i] = np.sum((res - true_effect) ** 2) / tot
        est_var[i] = np.sum((res - est_biased[i]) ** 2) / tot
    effect_naive_ave = np.sum(effect_naive) / tot


    file2write = open("ABtest_N_16000_T_100000.txt", 'w')
    print(mse_1, file=file2write)
    print('\n', file=file2write)
    print(mse_2, file=file2write)
    print('\n', file=file2write)
    print(mse_3, file=file2write)
    print('\n', file=file2write)
    print(est_var, file=file2write)
    print('\n', file=file2write)
    print(est_biased-true_effect, file=file2write)
    print('\n', file=file2write)
    print(effect_naive_ave-true_effect, file=file2write)
    print('\n', file=file2write)
    file2write.close()

