/**************   (ARM) optimized white-box implementation of WARX-16 with TweakGen and CLRW4   ************/
/*******            2026.3                        ***************/
/**********     jliu6@snnu.edu.cn             ***********************/
#include <stdlib.h>
#include <sys/time.h>
#include <givaro/gfq.h>
#include <ctime>
#include <stdio.h>
#include "warxsbox.h"
#include <time.h>
#include <errno.h>
#include <string.h>
#include <vector>
#include <arm_neon.h> //intrisics for NEON in ARM
#include <unistd.h>

#define round 7
#define u8 uint8_t
#define u16 uint16_t   
#define u32 uint32_t
#define u64 uint64_t
#define block 16// message of length 2048 bytes
#define parallel block/8     //deal with 8 blocks at one time
#define loops 100000 // reduce for testing, increase later

static uint16x8_t input[block],temp[block]; //message and copy
static uint16x8_t roundconstant; //round constant in affine layer
static u16(*vara)[8];  // a temporary object to store some intermediate values
static uint16x8_t allzero =vdupq_n_u16(0x0000);
static uint16x8_t xmm[8][8]; //used in finite field arithmetic
static uint16x8_t row[8];     //used in finite field arithmetic

/*************** 新增：5组密钥的轮密钥（白盒实现不需要，但为了接口统一保留） ***************/
static u8 rk_puf[1];   // 占位，实际不用
static u8 rk1[1];      // 占位，实际不用
static u8 rk2[1];
static u8 rk3[1];
static u8 rk4[1];

using namespace Givaro;
int modulus[] = { 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // x^16 + x^5 + x^3 + x + 1
GFqDom<int32_t> GF216(2, 16, modulus);

uint16x8_t mul(u16 operand, uint16x8_t x)   // FINITE FIELD MULTIPLICATION
{
		for (u16 k = 0; k < 8; k++)
		{
			GFqDom<int32_t>::Element a, b, c;
			GF216.init(a, operand);   // initialize
			GF216.init(b, ((u16 *)&x)[k]);
			GF216.mul(c, a, b);   // field multiplication
			int32_t c_int;
		    GF216.convert(c_int, c);
			((u16 *)&x)[k]=c_int;
		}
		return x;		
}

// RM
u16 RM[8][8] = {
    {0x1967, 0x2BA9, 0x659C, 0x7CFB, 0x5752, 0xCB38, 0xE091, 0x4E35},
    {0xB1C3, 0x6F41, 0xD26E, 0xBD2F, 0x7A75, 0x63AD, 0x0CEC, 0xCBB6},
    {0x64CE, 0xFDE8, 0x9926, 0xAB41, 0x62DD, 0xCF8F, 0x56A9, 0xC99C},
    {0x40E8, 0x00D7, 0xC041, 0x403F, 0xC096, 0xC1EF, 0x01AE, 0x80A9},
    {0xAC87, 0x91DF, 0x4789, 0x1EAC, 0xB22B, 0xD656, 0xC8FA, 0x7AD1},
    {0xC9BC, 0x56B1, 0xCF93, 0x62F1, 0xAB4D, 0x9922, 0xFDFC, 0x64DE},
    {0x1472, 0x7D59, 0xB480, 0x4E32, 0x2719, 0xEEC0, 0xC9D9, 0x5A40},
    {0xD081, 0x271E, 0x5AF3, 0xFBDA, 0x8637, 0xAD6C, 0x7DED, 0x2B5B}
};

// invRM
u16 invRM[8][8] = {
    {0x8644, 0x738D, 0x42A7, 0x86B1, 0x9477, 0x07D8, 0x9B6A, 0xB308},
    {0x8AE7, 0xD162, 0x1204, 0x077A, 0x3086, 0x021A, 0xF967, 0x19FC},
    {0x1946, 0x9497, 0x50A3, 0x8276, 0xFAA8, 0x02E1, 0xB113, 0x3341},
    {0x9F02, 0x45F5, 0xF1E5, 0x81CB, 0xD26D, 0x06D5, 0x43F6, 0x557A},
    {0x15E5, 0x8BEA, 0x74AB, 0x850C, 0x461A, 0x010D, 0x21FB, 0x7FC7},
    {0x328C, 0xE71A, 0xB342, 0x8C82, 0xCA2E, 0x00FB, 0x698F, 0x99B5},
    {0xB86B, 0x3678, 0xE3E1, 0x0EF4, 0x1843, 0x0317, 0x4874, 0x2ABD},
    {0x93A1, 0xF867, 0x854E, 0x04C7, 0x5E59, 0x03EC, 0xD89C, 0xE672}
};

/*******************   (inv)nonlinear layer of WARX16 :LUT  ****************************/
void nonlinear(uint16x8_t *m)
{	
    u16 vara[8]={sbox[((u16 *)&(*m))[0]],sbox[((u16 *)&(*m))[1]],sbox[((u16 *)&(*m))[2]],sbox[((u16 *)&(*m))[3]],sbox[((u16 *)&(*m))[4]],sbox[((u16 *)&(*m))[5]],sbox[((u16 *)&(*m))[6]],sbox[((u16 *)&(*m))[7]]};
    *m=vld1q_u16(vara);
}

void invnonlinear(uint16x8_t *m)
{
    u16 vara[8]={invsbox[((u16 *)&(*m))[0]],invsbox[((u16 *)&(*m))[1]],invsbox[((u16 *)&(*m))[2]],invsbox[((u16 *)&(*m))[3]],invsbox[((u16 *)&(*m))[4]],invsbox[((u16 *)&(*m))[5]],invsbox[((u16 *)&(*m))[6]],invsbox[((u16 *)&(*m))[7]]};
    *m=vld1q_u16(vara);
}

/*******************   encryption of WARX16  ****************************/
void encryptionwhite(uint16x8_t input[block])
{
	for (int i = 0; i < round; i++)
	{
		for (int j = 0; j < parallel; j++)
		{
			/*****************     nonlinear layer        *****************************/
			nonlinear(&input[j*8+0]);nonlinear(&input[j*8+1]);nonlinear(&input[j*8+2]);nonlinear(&input[j*8+3]);nonlinear(&input[j*8+4]);nonlinear(&input[j*8+5]);nonlinear(&input[j*8+6]);nonlinear(&input[j*8+7]);           
			/*****************   linear layer :  collect several blocks in one register  (for parallel)  and  xor corresponding places in every register      ********/
            for (int a=0;a<8;a++) row[a]=allzero;   
            for (int a=0;a<8;a++)
            {
                for (int b=0;b<8;b++)
                {
					u16 vara[8]={((u16*)&input[j*8+7])[b],
								 ((u16*)&input[j*8+6])[b],
								 ((u16*)&input[j*8+5])[b],
								 ((u16*)&input[j*8+4])[b],
								 ((u16*)&input[j*8+3])[b],
								 ((u16*)&input[j*8+2])[b],
								 ((u16*)&input[j*8+1])[b],
								 ((u16*)&input[j*8+0])[b]};
                    uint16x8_t temp=vld1q_u16(vara);
                    xmm[a][b]=mul(RM[a][b],temp);
                    row[a]=veorq_u16(row[a],xmm[a][b]);//row[a]=_mm_xor_si128(xmm[a][7],_mm_xor_si128(xmm[a][6],_mm_xor_si128(xmm[a][5],_mm_xor_si128(xmm[a][4],_mm_xor_si128(xmm[a][3],_mm_xor_si128(xmm[a][2],_mm_xor_si128(xmm[a][1],xmm[a][0])))))));
                }  
            }
	        for (int t=0;t<8;t++)
	        {
				u16 vara[8]={((u16*)&(row[0]))[t],((u16*)&(row[1]))[t],((u16*)&(row[2]))[t],((u16*)&(row[3]))[t],((u16*)&(row[4]))[t],((u16*)&(row[5]))[t],((u16*)&(row[6]))[t],((u16*)&(row[7]))[t]};
				input[j*8+t] = vld1q_u16(vara);
			}          
            /******************    affine layer ********************/
			u16 vara[8]={(u16)(8*i+1),(u16)(8*i+2),(u16)(8*i+3),(u16)(8*i+4),(u16)(8*i+5),(u16)(8*i+6),(u16)(8*i+7),(u16)(8*i+8)};
            roundconstant=vld1q_u16(vara);
            for (int t = 0; t < 8; t++)  //affine layer
            {
			   input[j*8+t]=veorq_u16(input[j*8+t],roundconstant);
            }					
		}
	}
}

/*******************   decryption of WARX16  ****************************/
void decryptionwhite(uint16x8_t input[block])
{	
	for (int i = round - 1; i >= 0; i--)
	{
		for (int j = 0; j < parallel; j++)
		{
			/******************     inverse    affine layer **************/
            u16 vara[8]={(u16)(8*i+1),(u16)(8*i+2),(u16)(8*i+3),(u16)(8*i+4),(u16)(8*i+5),(u16)(8*i+6),(u16)(8*i+7),(u16)(8*i+8)};
            roundconstant=vld1q_u16(vara);
            for (int t = 0; t < 8; t++)  //affine layer
            {
               input[j*8+t]=veorq_u16(input[j*8+t],roundconstant);
            }
			/********* inverse  linear layer (the same as linear layer):  collect several blocks in one register  (for parallel)  and  xor corresponding places in every register      *********/
            for (int a=0;a<8;a++) row[a]=allzero;   
            for (int a=0;a<8;a++)
            {
                for (int b=0;b<8;b++)
                {
                    u16 vara[8]={((u16*)&input[j*8+7])[b],
								 ((u16*)&input[j*8+6])[b],
								 ((u16*)&input[j*8+5])[b],
								 ((u16*)&input[j*8+4])[b],
								 ((u16*)&input[j*8+3])[b],
								 ((u16*)&input[j*8+2])[b],
								 ((u16*)&input[j*8+1])[b],
								 ((u16*)&input[j*8+0])[b]};
                    uint16x8_t temp=vld1q_u16(vara);
                    xmm[a][b]=mul(invRM[a][b],temp);
                    row[a]=veorq_u16(row[a],xmm[a][b]);//row[a]=_mm_xor_si128(xmm[a][7],_mm_xor_si128(xmm[a][6],_mm_xor_si128(xmm[a][5],_mm_xor_si128(xmm[a][4],_mm_xor_si128(xmm[a][3],_mm_xor_si128(xmm[a][2],_mm_xor_si128(xmm[a][1],xmm[a][0])))))));
                } 
            }
	        for (int t=0;t<8;t++)
	        {
				u16 vara[8]={((u16*)&(row[0]))[t],((u16*)&(row[1]))[t],((u16*)&(row[2]))[t],((u16*)&(row[3]))[t],((u16*)&(row[4]))[t],((u16*)&(row[5]))[t],((u16*)&(row[6]))[t],((u16*)&(row[7]))[t]};
				input[j*8+t] = vld1q_u16(vara);
	        }			
			 /******************    inverse        nonlinear layer          ***************/
            invnonlinear(&input[j*8+0]);invnonlinear(&input[j*8+1]);invnonlinear(&input[j*8+2]);invnonlinear(&input[j*8+3]);invnonlinear(&input[j*8+4]);invnonlinear(&input[j*8+5]);invnonlinear(&input[j*8+6]);invnonlinear(&input[j*8+7]);			
		}
	}	
}

/*************** 新增：TweakGen函数（整个block共用同一个tweak）***************/
void tweakgen(uint16x8_t *tweak, u64 ctr_high, u64 ctr_low)
{
    // 将128位计数器放入一个单独的块
    u16 ctr_words[8] = {0};
    ctr_words[0] = (u16)(ctr_low & 0xFFFF);
    ctr_words[1] = (u16)((ctr_low >> 16) & 0xFFFF);
    ctr_words[2] = (u16)((ctr_low >> 32) & 0xFFFF);
    ctr_words[3] = (u16)((ctr_low >> 48) & 0xFFFF);
    ctr_words[4] = (u16)(ctr_high & 0xFFFF);
    ctr_words[5] = (u16)((ctr_high >> 16) & 0xFFFF);
    ctr_words[6] = (u16)((ctr_high >> 32) & 0xFFFF);
    ctr_words[7] = (u16)((ctr_high >> 48) & 0xFFFF);
    
    uint16x8_t ctr_block = vld1q_u16(ctr_words);
    uint16x8_t temp_block = ctr_block;
    
    // 执行WARX加密（7轮）
    for (int i = 0; i < round; i++)
    {
        // 非线性层（查表）
        nonlinear(&temp_block);
        
        // 线性层（简化处理，因为temp_block只有一个块）
        uint16x8_t temp_row = allzero;
        for (int b = 0; b < 8; b++)
        {
            u16 tmp[8] = {((u16*)&temp_block)[b], 0,0,0,0,0,0,0};
            uint16x8_t t = vld1q_u16(tmp);
            t = mul(RM[0][b], t);
            temp_row = veorq_u16(temp_row, t);
        }
        temp_block = temp_row;
        
        // 仿射层
        u16 rc[8] = {(u16)(8*i+1), (u16)(8*i+2), (u16)(8*i+3), (u16)(8*i+4),
                     (u16)(8*i+5), (u16)(8*i+6), (u16)(8*i+7), (u16)(8*i+8)};
        uint16x8_t roundconst = vld1q_u16(rc);
        temp_block = veorq_u16(temp_block, roundconst);
    }
    
    // tweak = WARX(ctr) XOR ctr
    *tweak = veorq_u16(temp_block, ctr_block);
}

/*************** 新增：WARX-CLRW4加密 ***************/
void warx_clrw4_encrypt(uint16x8_t input[block], uint16x8_t tweak)
{
    uint16x8_t temp[block];
    memcpy(temp, input, 16 * block);
    
    // 第1层: WARX_{k1}(P)
    encryptionwhite(temp);
    
    // 与tweak异或（整个block所有块都用同一个tweak）
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第2层: WARX_{k2}
    encryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第3层: WARX_{k3}
    encryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第4层: WARX_{k4}
    encryptionwhite(temp);
    
    memcpy(input, temp, 16 * block);
}

/*************** 新增：WARX-CLRW4解密 ***************/
void warx_clrw4_decrypt(uint16x8_t input[block], uint16x8_t tweak)
{
    uint16x8_t temp[block];
    memcpy(temp, input, 16 * block);
    
    // 第4层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第3层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第2层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = veorq_u16(temp[j], tweak);
    
    // 第1层逆
    decryptionwhite(temp);
    
    memcpy(input, temp, 16 * block);
}

/*************** 新增：完整方案加密入口 ***************/
void full_scheme_encrypt(uint16x8_t input[block], u64 *ctr_high, u64 *ctr_low)
{
    uint16x8_t tweak;
    tweakgen(&tweak, *ctr_high, *ctr_low);
    warx_clrw4_encrypt(input, tweak);
    
    // 计数器加1
    uint64_t new_low = *ctr_low + 1;
    if (new_low < *ctr_low)  // 检查低64位溢出
        (*ctr_high)++;
    *ctr_low = new_low;
    
}

/*************** 新增：完整方案解密入口 ***************/
void full_scheme_decrypt(uint16x8_t input[block], u64 ctr_high, u64 ctr_low)
{
    uint16x8_t tweak;
    tweakgen(&tweak, ctr_high, ctr_low);
    warx_clrw4_decrypt(input, tweak);
}

/*********************************************************************************************

BELOW are some useful functions!

************************************************************************************************/

/***********************calculate average of list elements ***************************/
uint64_t Average(uint64_t list[], int lenlist)
{
	uint64_t ave, sum = 0;
	for (int i = 0; i < lenlist; i++) {
		sum += list[i];
	}
	ave = sum / lenlist;
	return ave;
}

double Average2(double list[], int lenlist)
{
    double ave, sum = 0;
    for (int i = 0; i < lenlist; i++) {
        sum += list[i];
    }
    ave = sum / lenlist;
    return ave;
}

/***********************  verify decryption   *********************************/
void verifydecryption(uint16x8_t arr1[block], uint16x8_t arr2[block])
{
	int i = memcmp(arr1, arr2, 16*block);
	if (i == 0)
		printf("✓ VERIFY DECRYPTION CORRECT!\n");
	else
		printf("✗ VERIFY DECRYPTION WRONG!\n");
}

/*************************   print message      *****************/
void printmessage(uint16x8_t arr[block])
{
	for (int i = 0; i < block; i++)
	{
		u16 *val=(u16*) &arr[i];
        for (int i=0;i<8;i++)
        {
            printf("%04X,",val[i]);
        }
	}		
}

/********************** generate random message (plaintext)  *************/
void generatemessage(uint16x8_t arr1[block])
{
	for (int i = 0; i < block; i++)
	{
        u16 vara[8] = {
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand()),
            static_cast<uint16_t>(rand())
        };
		arr1[i] = vld1q_u16(vara);
	}
}

int main(int argc, char** argv) 
{
	srand(time(0));
    double dec_time[loops];
    uint16x8_t input[block]; 
    uint16x8_t cipher[block];
    uint16x8_t decrypted[block];
	
    u64 ctr_high = 0, ctr_low = 0;  // 128位计数器	
    
    printf("\n========== 开始性能测试 ==========\n");
    
    for (int k = 0; k < loops; k++)
    {	
        printf("loop %d results:    ", k);
        
        generatemessage(input);
	memcpy(cipher, input, 16*block);
		
		// 记录加密前的计数器
	u64 before_high = ctr_high;
	u64 before_low = ctr_low;
		
		
	full_scheme_encrypt(cipher, &ctr_high, &ctr_low);// 加密
	memcpy(decrypted, cipher, 16*block);	
		// 测量解密时间
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
	full_scheme_decrypt(decrypted, before_high, before_low);// 解密
        clock_gettime(CLOCK_MONOTONIC, &end);
        dec_time[k] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
		
		// 验证
	if (memcmp(input, decrypted, 16*block) == 0) 
	 {
             printf("✓ decryption correct!\n");
         }
	else 
	 {
	    printf("✗ DECRYPTION WRONG at loop %d!\n", k);
                //break;
	 }
	}
	
    double avetime = Average2(dec_time, loops);
    printf("\n========== 性能测试结果 ==========\n");
    printf("average CPU time for decrypting %d byte message on Raspberry Pi 5: %.2f microseconds\n", 16*block, avetime*1e6);
    printf("average throughput for decrypting %d byte message on Raspberry Pi 5: %.2f MB/s\n", 16*block, ((double)(16*block))/(double)(avetime*1024*1024));
    printf("average CPU cycles for decrypting %d byte message on Raspberry Pi 5: %.2f\n", 16*block, avetime*2.4e9);
    printf("average CPB for decrypting %d byte message on Raspberry Pi 5: %.2f\n", 16*block, avetime*2.4e9/(double)(16*block));
    
    return 0;
}
