import galois
import numpy as np
import random

# 设置随机种子为当前时间，确保每次运行不同
random.seed()

def generate_random_gf16_elements(n=8):
    """生成n个GF(2^16)中的随机非零元素"""
    elements = []
    for _ in range(n):
        # 生成1到65535之间的随机数（避免0）
        val = random.randint(1, 0xFFFF)
        elements.append(val)
    return elements

def check_mds(matrix):
    """检查矩阵是否为MDS矩阵（所有2x2子矩阵满秩）"""
    n = len(matrix)
    all_full_rank = True
    for i1 in range(n):
        for i2 in range(i1+1, n):
            for j1 in range(n):
                for j2 in range(j1+1, n):
                    # 计算2x2子矩阵的行列式
                    det = matrix[i1][j1] * matrix[i2][j2] + matrix[i1][j2] * matrix[i2][j1]
                    if det == 0:
                        print(f"✗ 发现2x2子矩阵行列式为0: 行({i1},{i2}), 列({j1},{j2})")
                        all_full_rank = False
    return all_full_rank

def format_matrix_for_c(matrix, name):
    """将矩阵格式化为C代码"""
    print(f"\n// {name}")
    print(f"u16 {name}[8][8] = {{")
    for i in range(8):
        # 将galois域元素转换为整数，并取低16位
        row = [f"0x{int(matrix[i][j]) & 0xFFFF:04X}" for j in range(8)]
        print("    {" + ", ".join(row) + "}" + ("," if i < 7 else ""))
    print("};")

def main():
    print("=" * 70)
    print("GF(2^16) MDS矩阵生成器")
    print("=" * 70)
    
    # 定义GF(2^16) with irreducible polynomial x^16 + x^5 + x^3 + x + 1
    poly = galois.Poly.Degrees([16, 5, 3, 1, 0])  # x^16 + x^5 + x^3 + x + 1
    GF = galois.GF(2**16, irreducible_poly=poly)
    
    print(f"\n不可约多项式: {poly}")
    print(f"GF特征: {GF.characteristic}")
    print(f"GF阶数: {GF.order}")
    
    # 生成8个随机非零元素
    r = generate_random_gf16_elements(8)
    r_gf = GF(r)
    
    print(f"\n随机数向量 r = {[hex(x) for x in r]}")
    
    # KHAZAD MDS矩阵
    MP_values = [
        [0x01, 0x03, 0x04, 0x05, 0x06, 0x08, 0x0B, 0x07],
        [0x03, 0x01, 0x05, 0x04, 0x08, 0x06, 0x07, 0x0B],
        [0x04, 0x05, 0x01, 0x03, 0x0B, 0x07, 0x06, 0x08],
        [0x05, 0x04, 0x03, 0x01, 0x07, 0x0B, 0x08, 0x06],
        [0x06, 0x08, 0x0B, 0x07, 0x01, 0x03, 0x04, 0x05],
        [0x08, 0x06, 0x07, 0x0B, 0x03, 0x01, 0x05, 0x04],
        [0x0B, 0x07, 0x06, 0x08, 0x04, 0x05, 0x01, 0x03],
        [0x07, 0x0B, 0x08, 0x06, 0x05, 0x04, 0x03, 0x01]
    ]
    
    MP = GF(MP_values)
    
    # 构建对角矩阵 diag(r)
    diag_r = GF(np.diag(r))
    
    # 计算 MR = diag(r) * MP
    MR = diag_r @ MP
    
    print("\n" + "=" * 70)
    print("MR 矩阵 (加密矩阵)")
    print("=" * 70)
    print("MR = diag(r) × MP")
    print()
    for i in range(8):
        # 转换为整数并取低16位显示
        row_hex = [f"0x{int(MR[i][j]) & 0xFFFF:04X}" for j in range(8)]
        print("{" + ", ".join(row_hex) + "},")
    
    # 计算 MR 的逆矩阵
    MR_inv = np.linalg.inv(MR)
    
    print("\n" + "=" * 70)
    print("MR 的逆矩阵 (解密矩阵)")
    print("=" * 70)
    for i in range(8):
        row_hex = [f"0x{int(MR_inv[i][j]) & 0xFFFF:04X}" for j in range(8)]
        print("{" + ", ".join(row_hex) + "},")
    
    # 验证 MR * MR_inv = I
    print("\n" + "=" * 70)
    print("验证 MR × MR_inv = I")
    print("=" * 70)
    
    I_check = MR @ MR_inv
    is_identity = True
    for i in range(8):
        for j in range(8):
            expected = 1 if i == j else 0
            if I_check[i][j] != expected:
                is_identity = False
                print(f"✗ 位置 ({i},{j}): 期望 0x{expected:04X}, 实际 0x{int(I_check[i][j]) & 0xFFFF:04X}")
    
    if is_identity:
        print("✓ 验证通过: MR × MR_inv = I")
        
        # 也验证 MR_inv × MR = I
        I_check2 = MR_inv @ MR
        is_identity2 = True
        for i in range(8):
            for j in range(8):
                expected = 1 if i == j else 0
                if I_check2[i][j] != expected:
                    is_identity2 = False
        
        if is_identity2:
            print("✓ 验证通过: MR_inv × MR = I")
    else:
        print("\n✗ 验证失败，矩阵求逆不正确")
    
    # 检查MDS性质
    print("\n" + "=" * 70)
    print("检查矩阵的MDS性质")
    print("=" * 70)
    
    if check_mds(MR):
        print("✓ MR 是MDS矩阵")
    else:
        print("✗ MR 不是MDS矩阵")
    
    # 生成C代码格式
    print("\n" + "=" * 70)
    print("C代码格式的矩阵定义")
    print("=" * 70)
    
    format_matrix_for_c(MR, "RM")
    format_matrix_for_c(MR_inv, "invRM")

if __name__ == "__main__":
    main()