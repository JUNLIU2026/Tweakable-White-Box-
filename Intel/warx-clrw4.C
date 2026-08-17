/**************    optimized white-box implementation of WARX-16   ************/
/*******            2020.1                          ***************/
/**********     junjunll1212@gmail.com              ***********************/

#include <stdlib.h>
#include<sys/time.h>
#include <givaro/gfq.h>
#include <ctime>
#include <stdio.h>
#include "warxsbox.h"
#include<time.h>
#include<errno.h>
#include<string.h>
#include<vector>
#include <stdint.h>     //for int8_t
#include <string.h>     //for memcmp
#include <emmintrin.h>  // intrinsics for SSE2
#include <tmmintrin.h>
#include <immintrin.h>   //intrinsics for AVX2
#include <avx2intrin.h>

#define round 7
#define u8 uint8_t
#define u16 uint16_t   
#define u32 uint32_t
#define u64 uint64_t
#define block 128// message of length 2048 bytes
#define parallel block/8     //deal with 8 blocks at one time
#define loops 100000 //  repeat 100000 times

#ifdef __GNUC__
#include <x86intrin.h>
#endif
#ifdef _MSC_VER_
#include <intrin.h>
#endif

static __m128i input[block],temp[block];   // message and copy message
static __m128i allzero  =_mm_set_epi8 (0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00); 
static __m128i MSB8_m= _mm_set_epi16(0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000,0x0000);//used in finite field arithmetic
static __m128i xmm[8][8]; //used in finite field arithmetic
static __m128i row[8];     //used in finite field arithmetic

using namespace Givaro;
int modulus[] = { 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }; // x^16 + x^5 + x^3 + x + 1
GFqDom<int32_t> GF216(2, 16, modulus);


__m128i mul(u16 operand, __m128i x)   // FINITE FIELD MULTIPLICATION
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
void nonlinear(__m128i *m)
{
	*m=_mm_set_epi16(sbox[((u16 *)&(*m))[7]],sbox[((u16 *)&(*m))[6]],sbox[((u16 *)&(*m))[5]],sbox[((u16 *)&(*m))[4]],sbox[((u16 *)&(*m))[3]],sbox[((u16 *)&(*m))[2]],sbox[((u16 *)&(*m))[1]],sbox[((u16 *)&(*m))[0]]);
}

void invnonlinear(__m128i *m)
{
	*m=_mm_set_epi16(invsbox[((u16 *)&(*m))[7]],invsbox[((u16 *)&(*m))[6]],invsbox[((u16 *)&(*m))[5]],invsbox[((u16 *)&(*m))[4]],invsbox[((u16 *)&(*m))[3]],invsbox[((u16 *)&(*m))[2]],invsbox[((u16 *)&(*m))[1]],invsbox[((u16 *)&(*m))[0]]);
}

/*******************   encryption of WARX16  ****************************/
void encryptionwhite(__m128i input[block])
{
	for (int i = 0; i < round; i++)
	{
		for (int j = 0; j < parallel; j++)
		{
			/****************  AVX  nonlinear layer begins     ***********************/	
			for (int t=0;t<8;t++)
			{nonlinear(&input[j*8+t]);}
		/**********   linear layer begins:  collect several blocks in one register  (for parallel)  and  xor corresponding places in every register      *********/
            for (int a=0;a<8;a++)
            {
				row[a]=allzero;
                for (int b=0;b<8;b++)
                {
                            __m128i temp=_mm_set_epi16(
								((u16*)&input[j*8+7])[b],
								((u16*)&input[j*8+6])[b],
								((u16*)&input[j*8+5])[b],
								((u16*)&input[j*8+4])[b],
								((u16*)&input[j*8+3])[b],
								((u16*)&input[j*8+2])[b],
								((u16*)&input[j*8+1])[b],
								((u16*)&input[j*8+0])[b]
								);
                            xmm[a][b]=mul(RM[a][b],temp);
                            row[a]=_mm_xor_si128(row[a],xmm[a][b]);//row[a]=_mm_xor_si128(xmm[a][7],_mm_xor_si128(xmm[a][6],_mm_xor_si128(xmm[a][5],_mm_xor_si128(xmm[a][4],_mm_xor_si128(xmm[a][3],_mm_xor_si128(xmm[a][2],_mm_xor_si128(xmm[a][1],xmm[a][0])))))));
                }
            }
	        for (int t=0;t<8;t++)
	        {
				input[j*8+t]=_mm_set_epi16(((u16*)&(row[7]))[t],((u16*)&(row[6]))[t],((u16*)&(row[5]))[t],((u16*)&(row[4]))[t],((u16*)&(row[3]))[t],((u16*)&(row[2]))[t],((u16*)&(row[1]))[t],((u16*)&(row[0]))[t]);
	        }
        /****************    affine layer begins     ***************************/
		    __m128i roundconstant=_mm_set_epi16((u16)(8*i+8),(u16)(8*i+7),(u16)(8*i+6),(u16)(8*i+5),(u16)(8*i+4),(u16)(8*i+3),(u16)(8*i+2),(u16)(8*i+1));
			for (int t = 0; t < 8; t++)  //affine layer
            {input[j*8+t] ^= roundconstant;}			
		}
	}
}

/*******************   decryption of WARX16  ****************************/
void decryptionwhite(__m128i input[block]) 
{	
	for (int i = round-1; i>=0; i--)
	{
		for (int j = 0; j < parallel; j++)
		{
			/*************       inverse affine layer   begins   *************v******/
			__m128i roundconstant=_mm_set_epi16((u16)(8*i+8),(u16)(8*i+7),(u16)(8*i+6),(u16)(8*i+5),(u16)(8*i+4),(u16)(8*i+3),(u16)(8*i+2),(u16)(8*i+1));
            for (int t = 0; t < 8; t++)  
            {input[j*8+t] ^= roundconstant;} 
		/***  inverse linear layer  begins:  collect several blocks in one register  (for parallel)  and  xor corresponding places in every register   *********/ 
            for (int a=0;a<8;a++)
            {
				row[a]=allzero; 
                for (int b=0;b<8;b++)
                {
                            __m128i temp=_mm_set_epi16(
								((u16*)&input[j*8+7])[b],
								((u16*)&input[j*8+6])[b],
								((u16*)&input[j*8+5])[b],
								((u16*)&input[j*8+4])[b],
								((u16*)&input[j*8+3])[b],
								((u16*)&input[j*8+2])[b],
								((u16*)&input[j*8+1])[b],
								((u16*)&input[j*8+0])[b]
								);
                            xmm[a][b]=mul(invRM[a][b],temp);
                            row[a]=_mm_xor_si128(row[a],xmm[a][b]);//row[a]=_mm_xor_si128(xmm[a][7],_mm_xor_si128(xmm[a][6],_mm_xor_si128(xmm[a][5],_mm_xor_si128(xmm[a][4],_mm_xor_si128(xmm[a][3],_mm_xor_si128(xmm[a][2],_mm_xor_si128(xmm[a][1],xmm[a][0])))))));
                }  
            }
	        for (int t=0;t<8;t++)
	        {
				input[j*8+t]=_mm_set_epi16(((u16*)&(row[7]))[t],((u16*)&(row[6]))[t],((u16*)&(row[5]))[t],((u16*)&(row[4]))[t],((u16*)&(row[3]))[t],((u16*)&(row[2]))[t],((u16*)&(row[1]))[t],((u16*)&(row[0]))[t]);
	        }
		/******************    inverse nonlinear layer begins     ***************************/
		    for (int t=0;t<8;t++)
		    {invnonlinear(&input[j*8+t]);}
		}
	}	
}

/*************** TweakGen函数（整个block共用同一个tweak）***************/
void tweakgen(__m128i *tweak, u64 ctr_high, u64 ctr_low)
{
    // 将128位计数器放入一个单独的块
    __m128i ctr_block;
    u16 ctr_words[8] = {0};
    ctr_words[0] = (u16)(ctr_low & 0xFFFF);
    ctr_words[1] = (u16)((ctr_low >> 16) & 0xFFFF);
    ctr_words[2] = (u16)((ctr_low >> 32) & 0xFFFF);
    ctr_words[3] = (u16)((ctr_low >> 48) & 0xFFFF);
    ctr_words[4] = (u16)(ctr_high & 0xFFFF);
    ctr_words[5] = (u16)((ctr_high >> 16) & 0xFFFF);
    ctr_words[6] = (u16)((ctr_high >> 32) & 0xFFFF);
    ctr_words[7] = (u16)((ctr_high >> 48) & 0xFFFF);
    
    ctr_block = _mm_set_epi16(ctr_words[7], ctr_words[6], ctr_words[5], ctr_words[4],
                               ctr_words[3], ctr_words[2], ctr_words[1], ctr_words[0]);
    
    // 临时变量存储加密结果
    __m128i temp_block = ctr_block;
    
    // 执行WARX加密（7轮）
    for (int i = 0; i < round; i++)
    {
        // 非线性层（查表）
        nonlinear(&temp_block);
        
        // 线性层（这里简化处理，因为temp_block只有一个块）
        __m128i temp_row = allzero;
        for (int b = 0; b < 8; b++)
        {
            __m128i temp = _mm_set_epi16(
                ((u16*)&temp_block)[b], 0,0,0,0,0,0,0);
            temp = mul(RM[0][b], temp);
            temp_row = _mm_xor_si128(temp_row, temp);
        }
        temp_block = temp_row;
        
        // 仿射层
        __m128i rc = _mm_set_epi16(
            (u16)(8*i+8), (u16)(8*i+7), (u16)(8*i+6), (u16)(8*i+5),
            (u16)(8*i+4), (u16)(8*i+3), (u16)(8*i+2), (u16)(8*i+1));
        temp_block = _mm_xor_si128(temp_block, rc);
    }
    
    // tweak = WARX(ctr) XOR ctr
    *tweak = _mm_xor_si128(temp_block, ctr_block);
}

/*************** WARX-CLRW4加密 ***************/
void warx_clrw4_encrypt(__m128i input[block], __m128i tweak)
{
    __m128i temp[block];
    memcpy(temp, input, 16 * block);
    
    // 第1层: WARX_{k1}(P) - 注意这里用的是同一个encryptionwhite
    // 但我们没有不同的密钥，所以实际上每层都一样（简化版本，仅用于测试效率）
    encryptionwhite(temp);
    
    // 与tweak异或（整个block所有块都用同一个tweak）
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第2层: WARX_{k2}
    encryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第3层: WARX_{k3}
    encryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第4层: WARX_{k4}
    encryptionwhite(temp);
    
    memcpy(input, temp, 16 * block);
}

/*************** WARX-CLRW4解密 ***************/
void warx_clrw4_decrypt(__m128i input[block], __m128i tweak)
{
    __m128i temp[block];
    memcpy(temp, input, 16 * block);
    
    // 第4层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第3层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第2层逆
    decryptionwhite(temp);
    for (int j = 0; j < block; j++)
        temp[j] = _mm_xor_si128(temp[j], tweak);
    
    // 第1层逆
    decryptionwhite(temp);
    
    memcpy(input, temp, 16 * block);
}

/*************** 完整方案加密入口 ***************/
void full_scheme_encrypt(__m128i input[block], u64 *ctr_high, u64 *ctr_low)
{
    __m128i tweak;
    tweakgen(&tweak, *ctr_high, *ctr_low);  
    warx_clrw4_encrypt(input, tweak);
    
    // 计数器加1
    uint64_t new_low = *ctr_low + 1;
    if (new_low < *ctr_low)  // 检查低64位溢出
        (*ctr_high)++;
    *ctr_low = new_low;
    
}

/*************** 完整方案解密入口 ***************/
void full_scheme_decrypt(__m128i input[block], u64 ctr_high, u64 ctr_low)
{
    __m128i tweak;
    tweakgen(&tweak, ctr_high, ctr_low);  
    warx_clrw4_decrypt(input, tweak);
}

/*********************************************************************************************

BELOW are some useful functions!

************************************************************************************************/
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

void print128_num(__m128i var)
{
	  u16 *val=(u16*) &var;
      for (int i=0;i<8;i++)
      {
            printf("%04X",val[i]);
      }
}

void printmessage(__m128i arr[block])
{
	for (int i = 0; i < block; i++)
	{
		print128_num(arr[i]);
	}	
}

/********************** generate random message (plaintext)  ************************************/
void generatemessage(__m128i arr1[block])
{
	for (int i = 0; i < block; i++)
	{
		arr1[i] = _mm_set_epi16(rand(),rand(),rand(),rand(),rand(),rand(),rand(),rand());
	}
}

/***********************  verify decryption   **********************************/
void verifydecryption(__m128i arr1[block], __m128i arr2[block])
{
	int i = memcmp(arr1, arr2, 16*block);
	if (i == 0)
		printf("✓ VERIFY DECRYPTION CORRECT!");
	else
		printf("✗ VERIFY DECRYPTION WRONG!");
}

int main(int argc, char** argv) 
{
	srand(time(0));
    double time[loops];
	__m128i input[block]; 
    __m128i cipher[block];
    __m128i decrypted[block];
	
	u64 ctr_high = 0, ctr_low = 0;  // 128位计数器
	
    
    printf("\n========== 开始性能测试 ==========\n");  
	for (int k = 0; k < loops; k++)
	{	
		printf("********* loop %d results below  ************\n", k);		
		generatemessage(input);
		memcpy(cipher, input, 16*block);
		
		// 记录加密前的计数器
		u64 before_high = ctr_high;
		u64 before_low = ctr_low;
		
		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start); 
		
		// 完整方案加密
		full_scheme_encrypt(cipher, &ctr_high, &ctr_low);
		
		clock_gettime(CLOCK_MONOTONIC, &end); 
    	time[k] = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) /1e9;
		
		// 解密验证
        memcpy(decrypted, cipher, 16*block);
        full_scheme_decrypt(decrypted, before_high, before_low);
        
        if (memcmp(input, decrypted, 16*block) == 0)
            printf("✓ 解密正确");
        else
            printf("✗ 解密错误");
		printf("\n");
	}
	
	double avetime = Average2(time, loops);
    printf("\n========== 性能测试结果 ==========\n");
    printf("average CPU time for encrypting %d byte message: %.2f microseconds\n", 16*block, avetime*1e6);
    printf("average throughput: %.2f MB/S \n", (double)(16*block)/(double)(avetime*1024*1024));
    printf("average CPU cycles: %.2f \n", avetime*3.4e9);
    printf("average CPB: %.2f \n", avetime*3.4e9/(double)(16*block));
    
	return 0;	
}