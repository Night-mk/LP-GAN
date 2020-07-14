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
    batchsize = 1000
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

    return time
    
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
    num = 1
    bit_length = 32
    # u = np.abs(np.random.randn(batchsize,num,2,2))
    u = np.array([[[[0.00828291,0.00256],[0.00742,0.0002598]]]])
    # u_32 = np.floor_divide(u*(2**(bit_length//2)), 1)
    # u = np.random.randint(0, 2**8, size=(batchsize,num))
    
    u1 = np.random.randn(batchsize,num,2,2)
    # u1_32 = np.floor_divide(u1*(2**(bit_length//2)), 1)
    # u1 = np.random.randint(-2**8, 2**8, size=(batchsize,num))
    u2 = u-u1
    print('u: \n', u)
    print('u1: \n', u1)
    print('u2: \n', u2)
    # u2_32 = u_32 - u1_32
    start_time = time.time()
    f1, f2 = secp.SSqrt(u1+1e-5, u2, 5, inverse_required=True, bit_length=bit_length)
    # f1, f2 = secp.SSqrt(u1_32, u2_32, 5, inverse_required=False, bit_length=bit_length)
    end_time = time.time()
    print('time consume: \n',(end_time-start_time)*1000)
    # print('f: \n', (f1+f2))
    # print('sqrt err: \n', (f1+f2)-np.sqrt(u))
    # print('u.sqrt: \n', np.sqrt(u))
    # print('sqrt err: \n', (f1+f2)/(2**(bit_length//4))-np.sqrt(u))
    print('sqrt err: \n', f1+f2-1/np.sqrt(u+1e-5))
    # print('sqrt err mean: \n', np.mean(f1+f2-1/np.sqrt(u)))


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


def matrix_transform_test():
    # 将二维数组进行横向复制   
    a = np.random.randn(3,4) # a=[3,4]
    # a = np.ones((5,25)) # a=[3,4]
    b = np.random.randn(4,3) # b=[4,3]
    # b = np.ones((25,169)) # b=[4,3]

    a_shape = a.shape
    b_shape = b.shape

    # start_dot_time = time.time()
    # np.dot(a,b)
    # end_dot_time = time.time()
    # print('dot time consume: ', (end_dot_time-start_dot_time)*1000)

    print('a: \n', a)
    print('b: \n', b)
    # print('b.shape[1]: \n', b.shape[1])
    
    start_time = time.time()
    a_scale = np.tile(a, b_shape[1])
    b_scale = np.tile(b.T.reshape(-1), (a_shape[0],1))
    # print('a_scale: \n', a_scale)
    # print('b_scale: \n', b_scale)
    # print('a_scale.shape: \n', a_scale.shape)
    # print('b_scale.shape: \n', b_scale.shape)

    a_b_scale = a_scale+b_scale
    a_b_resize = np.zeros((a_shape[0], b_shape[1]))
    for i in range(0, a_shape[0]*b_shape[0], a_shape[1]):
        # print(i)
        # print(i+a_shape[1])
        # print(a_b_scale[:, i:i+a_shape[1]])
        a_b_resize[:, i//a_shape[0]] = np.sum(a_b_scale[:, i:i+a_shape[1]], axis=1)
    end_time = time.time()
    print('time consume: ', (end_time-start_time)*1000)
    # print('a_b_scale: \n',a_b_scale)
    print('a_b_resize: \n',a_b_resize)

def secmul_dot_test():
    a = np.random.randn(5,25) # a=[3,4]
    b = np.random.randn(25,50) # b=[4,3]
    # print('a: \n', a)
    # print('b: \n', b)
    a1 = np.ones(a.shape)
    b1 = np.ones(b.shape)
    a2 = a-a1
    b2 = b-b1

    ab_dot = np.dot(a,b)
    
    # f1, f2 = secp.SecMul_dot(a1, b1, a2, b2)
    # f1, f2 = secp.SecMul_dot_2(a1, b1, a2, b2)
    f1, f2 = secp.SecMul_dot_3(a1, b1, a2, b2)

    # print('ab_dot: \n', ab_dot)
    # print('f1+f2: \n', f1+f2)
    print('error : \n', (f1+f2)-ab_dot)

def secConvSpeed_test():
    
    a = np.random.randn(5,5) # a=[3,4]
    b = np.random.randn(5,5) # b=[4,3]
    a1 = np.random.randn(5,5)
    b1 = np.random.randn(5,5)
    a2 = a-a1
    b2 = b-b1
    start_time = time.time()
    for i in range(0, 5):
        for j in range(0, 169):
            f1, f2 = secp.SecMul_matrix(a1,b1,a2,b2, bit_length=32)
    end_time = time.time()
    print('conv time: ', (end_time-start_time)*1000)


def bit_extract_test():
    # a = np.random.randn(2,2) # a=[3,4]
    # a = np.random.randn(128,100) # a=[3,4]
    # a = np.random.randn(128,5,14,14) # a=[3,4]
    a = np.random.randn(64,5,14,14) # a=[3,4]
    a_shape = a.shape
    bit_length=64
    start_time = time.time()
    ### 对比bin_repr和bin的效率
    # print(len(bin(10)[2:]))
    num_bin = secp.DFC(a, bit_length)
    # a_int = np.array(a*(2**(bit_length//2))).astype(np.int64)
    # s = np.zeros(a_int.shape).astype(np.int) #一定要转为int类型
    # s[a_int<0] = 1
    # s = s.astype(np.str).reshape(-1)
    # num_bin_shape = list(a_int.shape)
    # num_bin_shape.append(bit_length)
    # a_int_list = abs(a_int.reshape(-1)).tolist()
    # ## bin_repr = [np.binary_repr(x, 64-1) for x in a_int_list]
    '''
    bin_repr_num = ''
    bin_num = ''
    start_bin_repr=time.time()
    for index, x in enumerate(a_int_list):
        bin_repr_num += np.binary_repr(x, 64-1)
    end_bin_repr=time.time()
    start_bin = time.time()
    for index, x in enumerate(a_int_list):
        bin_x = bin(x)
        bin_x = '0'*(bit_length-1-len(bin_x))+bin_x
        bin_num+=bin_x
    end_bin = time.time()
    
    print('bin_repr_num: \n',bin_repr_num)
    print('bin_num: \n',bin_repr_num)
    '''
    # num_bin = np.array(list(bin_repr)).astype(np.int).reshape(num_bin_shape)
    # bin_repr = map(bin_repr_64, a_int_list)
    # print(type(a_int_list))

    end_time = time.time()
    print('time consume: ',(end_time-start_time)*1000)
    # print('time consume: ',(end_bin_repr-start_bin_repr)*1000)
    # print('time consume: ',(end_bin-start_bin)*1000)
    # print('a_int: ', len(a_int_list))
    # print('bin_repr: ',num_bin)
    print('num_bin_shape: ',num_bin.shape)
    # print('DFC num_bin_shape: ',num_bin.shape)

# def bin_repr_64(x):
    # return np.binary_repr(x, 64-1)

if __name__ == '__main__':
    # secmul_m_test()
    # secor_m_test()
    # secxor_m_test()
    # secmsb_m_test()
    # secmul_m_test_fix()
    '''
    avg_time = 0
    for i in range(10):
        time = seccmp_m_test()
        avg_time+=time
    print('avg_time: ', avg_time/10)
    '''
    # secrc_m_test()
    ssqrt_m_test()
    # matrix_transform_test()
    # secmul_dot_test()
    # secConvSpeed_test()
    # bit_extract_test()