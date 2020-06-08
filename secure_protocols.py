'''
    secure_protocols.py 构建LP-GAN使用的基础安全协议
'''

import random
import numpy as np
from numpy import ndarray
import math

################################  线性计算  ################################

'''安全加法协议 SecAdd'''

def SecAdd(u1,v1,u2,v2):
	f1 = u1 + v1
	f2 = u2 + v2
	return f1,f2

'''安全乘法协议 SecMul'''
def SecMul(u1,v1,u2,v2):
    a1,a2,b1,b2,c1,c2 = mul_generate_random()
    # print((a1+a2)*(b1+b2),",",(c1+c2))

    alpha1 = u1 - a1
    beta1 = v1 - b1
    alpha2 = u2 - a2
    beta2 = v2 - b2

    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + b1 * S1_alpha + a1 * S1_beta

    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + b2 * S2_alpha + a2 * S2_beta + S2_alpha * S2_beta

    return f1,f2

# 生成随机值
def mul_generate_random():
    a = random.randint(10,20)
    b = random.randint(10,20)
    c = a * b 
    #1
    a1 = random.randint(5,a-5)
    a2 = a - a1
    b1 = random.randint(5,b-5)
    b2 = b - b1
    c1 = random.randint(5,c-5)
    c2 = c - c1

    return a1,a2,b1,b2,c1,c2

'''安全乘法协议（矩阵形式）SecMul_matrix'''
def SecMul_matrix(u1,v1,u2,v2):
    a1,a2,b1,b2,c1,c2 = mul_generate_random()
    print((a1+a2)*(b1+b2),",",(c1+c2))

    # S1
    alpha1 = u1 - a1
    beta1 = v1 - b1
    # S2
    alpha2 = u2 - a2
    beta2 = v2 - b2

    # S1
    S1_alpha = alpha1 + alpha2
    S1_beta = beta1 + beta2
    f1 = c1 + b1 * S1_alpha + a1 * S1_beta
    # S2
    S2_alpha = alpha1 + alpha2
    S2_beta = beta1 + beta2
    f2 = c2 + b2 * S2_alpha + a2 * S2_beta + np.multiply(S2_alpha, S2_beta) 

    return f1,f2

def test_secmul():
    x1 = 5


################################  非线性计算  ################################
'''
SecOr
'''
def SecOr(u1,v1,u2,v2):
    m1,m2 = SecMul(-u1,v1,-u2,v2)
    
    # S1
    f1 = u1+v1+m1
    f2 = u2+v2+m2
    return f1, f2

'''
SecXor
'''
def SecXor(u1,v1,u2,v2):
    m1,m2 = SecMul(-2*u1,v1,-2*u2,v2)
    
    # S1
    f1 = u1+v1+m1
    f2 = u2+v2+m2
    return f1, f2


################################  比较算法  ################################

# precision>=16
# 10进制移位计算
# def float2Int(num, precision=16):
#     num_int = int(math.floor(num*(10**precision)))
#     return num_int

def float2bin(num, precision=16):
    bit_length = 32
    # check the symbol
    s = 0
    if num<0: s=1

    # transform it into positive integer
    num = abs(num)
    # break the number into integer and decimal part
    integer = int(num)
    decimal = num-integer
    # convert integer part to bin
    integer_binary = bin(integer)[2:]
    # integer_binary1 = bin(integer)
    # print("integer part: ", integer_binary1)
    # print("integer part: ", integer_binary)

    # convert binary part to bin
    decimal_binary = dec2bin(decimal, precision)
    
    # fill 0 if needed, handle symbol
    num_bin = integer_binary+decimal_binary
    length = len(num_bin)

    if length>bit_length-1: 
        # if integer is too large, cut off the decimal part
        num_bin=num_bin[:bit_length-1]
    else:
        # if the number length is not enough
        while length<bit_length-1:
            num_bin='0'+num_bin
            length+=1
    
    num_bin=str(s)+num_bin
    # print("len: ",len(num_bin))

    return num_bin

'''
decimal->binary
dec2bin
'''
def dec2bin(dec, precision=16):
    r = 0
    binary =''
    while dec!=0:
        dec = dec*2
        binary+=str(int(dec))
        dec = dec-int(dec)
        r+=1
        if r==precision: break
    length = len(binary)
    while length<precision:
        binary+='0'
        length+=1

    return binary

'''
SecMSB
'''
def SecMSB(u1, u2, bit_length=32):
    L = bit_length
    t1 = ndarray((L+1,), int)
    t2 = ndarray((L+1,), int)
    f1 = ndarray((L,), int)
    f2 = ndarray((L,), int)
    zeta1 = 0
    zeta2 = 0

    # offline
    t1[L] = np.random.randint(-5, 5)
    t2[L] = 1-t1[L]
    # print("ori u1, u2:", u1, u2)

    # online
    for i in range(L-1, -1, -1):
        # print("t1,t2: ", t1[i+1], t2[i+1])
        # print("u1,u2: ", 1-int(u1[i]), -int(u2[i]))
        t1[i], t2[i] = SecMul(t1[i+1], 1-u1[i], t2[i+1], u2[i])

        # 避免迭代后的secMul爆炸
        if t1[i] > 10e3:
            t1[i]-=t1[i]
            t2[i]+=t1[i]
        if t1[i] < -10e3:
            t1[i]+=t1[i]
            t2[i]-=t1[i]
        if t2[i] > 10e3:
            t2[i]-=t2[i]
            t1[i]+=t2[i]
        if t2[i] < -10e3:
            t2[i]+=t2[i]
            t1[i]-=t2[i]
        # print("t1,t2: ", t1[i], t2[i])
    
    for i in range(L-1, -1, -1):
        # S1
        f1[i] = t1[i+1]-t1[i]
        # S2
        f2[i] = t2[i+1]-t2[i]

    for i in range(0, L):
        zeta1 += f1[i]
        zeta2 += f2[i]
    
    # print(f1)
    # print(f2)

    return f1,zeta1,f2,zeta2

def DFC(num, precision=16):
    binary = float2bin(num, precision)
    L = precision*2-1
    return binary[:L]

'''
SecCmp
'''
def SecCmp(u1,v1,u2,v2, bit_length=32):
    L = bit_length
    c1 = ndarray((L,), int)
    c2 = ndarray((L,), int)
    e1 = ndarray((L,), int)
    e2 = ndarray((L,), int)
    xi1 = 0
    xi2 = 0
    # S1
    a = u1-v1
    a_bin = float2bin(a)

    # S2
    b = v2-u2
    b_bin = float2bin(b)
    print("a,b: ", a,b)

    print("a_bin:", a_bin)
    print("b_bin:", b_bin)

    for i in range(0, L):
        c1[i], c2[i] = SecXor(int(a_bin[i]),int(b_bin[i]), 0,0)
    print("c1,c2: ", c1+c2)
    
    d1,zeta1,d2,zeta2 = SecMSB(c1,c2,L)
    print("d,zeta: ", d1+d2,zeta1+zeta2)
    
    for i in range(0, L):
        e1[i], e2[i] = SecMul(int(a_bin[i]), d1[i], 0, d2[i])
        
    print("e: ", e1+e2)

    for i in range(0, L):
        xi1 += e1[i]
        xi2 += e2[i]
    
    # print(int(a_bin[L-1]),int(b_bin[]))
    iota1, iota2 = SecOr(int(a_bin[0]),int(b_bin[0]), 0, 0)
    print("iota:", iota1+iota2)
    nu1, nu2 = SecXor(iota1, xi1, iota2, xi2)
    print("nu:", nu1+nu2)
    f1, f2 = SecMul(nu1, zeta1, nu2, zeta2)

    return f1, f2

def test():
    # u1=56
    # u2=23
    # v1=56
    # v2=6
    # add
    # f1,f2 = SecAdd(u1,v1,u2,v2)
    # print("f1:",f1,";f2:",f2,";result:",f1+f2,"real:",u1+v1+u2+v2)

    # mul
    # f1,f2 = SecMul(u1,v1,u2,v2)
    # print("f1:",f1,";f2:",f2,";result:",f1+f2,"real:",(u1+u2)*(v1+v2))

    # mul_matrix
    # u1=np.random.randint(-2**16, 2**16, (5,5))
    # u2=np.random.randint(-2**16, 2**16, (5,5))
    # v1=np.random.randint(-2**16, 2**16, (5,5))
    # v2=np.random.randint(-2**16, 2**16, (5,5))
    # f1,f2 = SecMul_matrix(u1,v1,u2,v2)
    # print("f1:",f1,"\n f2:",f2,"\n result:",f1+f2,"\n real:",np.multiply(u1+u2,v1+v2))


    # or
    # u = np.random.randint(0,2)
    # v = np.random.randint(0,2)
    # u1 = np.random.randint(-2**16, 2**16)
    # u2 = u-u1
    # v1 = np.random.randint(-2**16, 2**16)
    # v2 = v-v1
    # f1, f2 = SecOr(u1,v1,u2,v2)
    # print("f1:",f1,";f2:",f2,"u:",u,"v:",v,";result:",f1+f2,"real:", u or v)
    
    # xor
    # f1, f2 = SecXor(u1,v1,u2,v2)
    # print("f1:",f1,";f2:",f2,"u:",u,"v:",v,";result:",f1+f2)

    # cmp
    # dec = dec2bin(0.98954910, precision=16)
    # print(dec)
    # r = float2bin(98989.62265, 16)
    # print(r)
    
    # u1 = float2bin(78, 16)
    # u2 = float2bin(7889, 16)
    # u1 = DFC(78, 16)
    # u2 = DFC(7889, 16)
    # u2 = float2bin(996, 16)
    # print(u1,u2)
    # f1,zeta1,f2,zeta2 = SecMSB(u1, u2)
    # print(f1+f2, zeta1, zeta2, zeta1+zeta2)

    # u = np.random.randint(-2**16, 2**16)
    u = np.random.uniform(-2**16, 2**16)
    v = np.random.uniform(-2**16, 2**16)
    # v = np.random.randint(-2**16, 2**16)
    u1 = np.random.randint(10, 20)
    u2 = u-u1
    v1 = np.random.randint(10, 20)
    v2 = v-v1
    print("u,v:", u,v)
    f1, f2  = SecCmp(u1,v1,u2,v2)
    print(f1+f2)

if __name__ = '__main__':
    test_secmul()
