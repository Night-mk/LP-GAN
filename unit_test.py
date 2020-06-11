'''
    unit_test.py 用于单元测试隐私计算基础协议，隐私计算层的正确性等
'''

import secure_protocols as secp
import numpy as np
import time


'''
    测试：安全矩阵乘法协议 SecMul_Matrix（成功）
'''
def secmul_m_test():
    batchsize = 512
    u = np.random.randn(batchsize,4,3)
    v = np.random.randn(batchsize,4,3)
    # print('u: \n', u)
    # print('v: \n', v)
    u1 = np.random.randn(batchsize,4,3)
    u2 = u-u1
    v1 = np.random.randn(batchsize,4,3)
    v2 = v-v1
    
    start_time = time.time()
    f1, f2 = secp.SecMul_matrix(u1, v1, u2, v2)
    end_time = time.time()
    # print('u1: \n', u1)
    # print('u2: \n', u2)
    # print('v1: \n', v1)
    # print('v2: \n', v2)

    # print('cal err: \n', (f1+f2)-u*v)
    print('mul time: \n', end_time-start_time)

def secmul_m_test_fix():
    u1 = [[-56503, -56503, -56503, -56503],[-56503, -56503, -56503, -56503]]
    u2 = [[56504, 56504, 56504, 56504], [56504, 56504, 56504, 56504]]
    v1 = [[ 44795, -33891, -16734,  48148], [ 14328, -29446,  16009 , 27700]]
    v2 = [[-44795 , 33891,  16735, -48148], [-14327 , 29447, -16008, -27699]]

    u1 = np.array(u1)
    u2 = np.array(u2)
    v1 = np.array(v1)
    v2 = np.array(v2)

    f1, f2 = secp.SecMul_matrix(u1, v1, u2, v2)
    print('f1+f2: \n', (f1+f2))
    print('cal err: \n', (f1+f2)-(u1+u2)*(v1+v2))



'''
    测试：安全矩阵或，异或协议 SecOr,SecXor（成功）
'''
def secor_m_test():
    u = np.random.randint(0,2,size=(2,4))
    v = np.random.randint(0,2,size=(2,4))
    print('u: \n', u)
    print('v: \n', v)
    u1 = np.random.randn(2,4)
    u2 = u-u1
    v1 = np.random.randn(2,4)
    v2 = v-v1
    f1, f2 = secp.SecOr(u1, v1, u2, v2)
    print('cal err: \n', (f1+f2))

def secxor_m_test():
    u = np.random.randint(0,2,size=(2,4))
    v = np.random.randint(0,2,size=(2,4))
    print('u: \n', u)
    print('v: \n', v)
    u1 = np.random.randn(2,4)
    u2 = u-u1
    v1 = np.random.randn(2,4)
    v2 = v-v1
    f1, f2 = secp.SecXor(u1, v1, u2, v2)
    print('cal err: \n', (f1+f2))

'''
    测试：安全最重要比特协议 SecMSB
'''
def secmsb_m_test():
    L = 32
    u = np.random.randn(2,4)
    u_bin = secp.float2bin(u)
    u_arr_shape = list(u.shape)
    u_arr_shape.append(L)
    u_arr = secp.str2arr(u_bin, u_arr_shape)

    print('u: \n',u)
    print('u_arr: \n',u_arr)

    u1 = np.random.randint(-2**(L//2), 2**(L//2), size=u_arr_shape)
    u2 = u_arr - u1
    f1,zeta1,f2,zeta2 = secp.SecMSB(u1, u2)
    print('f1+f2: \n',f1+f2)
    print('zeta: \n',zeta1+zeta2)
    

'''
    测试：安全矩阵比较算法 SecCmp_Matrix（成功）
'''
def seccmp_m_test():
    batchsize = 10000
    num = 1
    bit_length = 32
    u = np.random.randn(batchsize,num)
    v = np.random.randn(batchsize,num)
    # print('u: \n', u)
    # print('v: \n', v)
    u1 = np.random.randn(batchsize,num)
    u2 = u-u1
    v1 = np.random.randn(batchsize,num)
    v2 = v-v1
    f1, f2, time = secp.SecCmp(u1, v1, u2, v2, bit_length)
    
    # print('f1+f2: \n',f1+f2)

'''
    测试：安全矩阵求根算法 SSqrt_Matrix
'''
# 测试SRC算法
def secrc_m_test():
    batchsize = 1
    num = 20
    bit_length = 32
    # u = np.abs(np.random.randn(batchsize,num))
    u = np.random.randint(0, 2**8, size=(batchsize,num))
    # print('u: \n', u)
    # u1 = np.random.randn(batchsize,num)
    u1 = np.random.randint(-2**8, 2**8, size=(batchsize,num))
    u2 = u-u1
    # print('u1: \n', u1)
    # print('u2: \n', u2)

    start_time = time.time()
    m1,m2,p = secp.SecRC(u1, u2, bit_length=32)
    end_time = time.time()
    print('p: \n', m1+m2)
    print('time consume: \n',(end_time-start_time)*1000)

def ssqrt_m_test():
    batchsize = 1
    num = 100
    bit_length = 32
    u = np.abs(np.random.randn(batchsize,num))
    # u = np.random.randint(0, 2**8, size=(batchsize,num))
    # print('u: \n', u)
    u1 = np.random.randn(batchsize,num)
    # u1 = np.random.randint(-2**8, 2**8, size=(batchsize,num))
    u2 = u-u1
    start_time = time.time()
    f1, f2 = secp.SSqrt(u1, u2, 5, inverse_required=True, bit_length=bit_length)
    end_time = time.time()
    print('time consume: \n',(end_time-start_time)*1000)
    # print('f: \n', f1+f2)
    # print('u.sqrt: \n', np.sqrt(u))
    print('sqrt err: \n', f1+f2-1/np.sqrt(u))
    print('sqrt err mean: \n', np.mean(f1+f2-1/np.sqrt(u)))


'''
    测试：安全矩阵log算法 SecLog_Matrix
'''
def seclog_m_test():
    print('a')

'''
    测试：安全矩阵exp算法 SecExp_Matrix
'''
def secexp_m_test():
    print('a')

'''
    测试：安全矩阵1/n算法 SecInv_Matrix
'''
def secinv_m_test():
    print('a')


if __name__ == '__main__':
    # secmul_m_test()
    # secor_m_test()
    # secxor_m_test()
    # secmsb_m_test()
    # secmul_m_test_fix()
    # for i in range(5):
    #     seccmp_m_test()
    # secrc_m_test()
    ssqrt_m_test()