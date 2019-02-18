// -ck modified kernel taken from Phoenix taken from poclbm, with aspects of
// phatk and others.
// Modified version copyright 2011-2013 Con Kolivas

// This file is taken and modified from the public-domain poclbm project, and
// we have therefore decided to keep it public-domain in Phoenix.

// kernel-interface: poclbm SHA256d

#ifdef VECTORS4
	typedef uint4 u;
#elif defined VECTORS2
	typedef uint2 u;
#else
	typedef uint u;
#endif

__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
__constant uint initH[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#ifdef BITALIGN
	#pragma OPENCL EXTENSION cl_amd_media_ops : enable
	#define rotr(x, y) amd_bitalign((u)x, (u)x, (u)y)
#else // BITALIGN
	#define rotr(x, y) rotate((u)x, (u)(32 - y))
#endif
#ifdef BFI_INT
	// Well, slight problem... It turns out BFI_INT isn't actually exposed to
	// OpenCL (or CAL IL for that matter) in any way. However, there is 
	// a similar instruction, BYTE_ALIGN_INT, which is exposed to OpenCL via
	// amd_bytealign, takes the same inputs, and provides the same output. 
	// We can use that as a placeholder for BFI_INT and have the application 
	// patch it after compilation.

	// This is the BFI_INT function
	#define ch(x, y, z) amd_bytealign(x, y, z)
	
	// Ma can also be implemented in terms of BFI_INT...
	#define Ma(x, y, z) amd_bytealign( (z^x), (y), (x) )

	// AMD's KernelAnalyzer throws errors compiling the kernel if we use
	// amd_bytealign on constants with vectors enabled, so we use this to avoid
	// problems. (this is used 4 times, and likely optimized out by the compiler.)
	#define Ma2(x, y, z) bitselect((u)x, (u)y, (u)z ^ (u)x)
#else // BFI_INT
	//GCN actually fails if manually patched with BFI_INT

	#define ch(x, y, z) bitselect((u)z, (u)y, (u)x)
	#define Ma(x, y, z) bitselect((u)x, (u)y, (u)z ^ (u)x)
	#define Ma2(x, y, z) Ma(x, y, z)
#endif

#define E0(x) (rotr(x,2)^rotr(x,13)^rotr(x,22))
#define E1(x) (rotr(x,6)^rotr(x,11)^rotr(x,25))
#define O0(x) (rotr(x,7)^rotr(x,18)^(x>>3U))
#define O1(x) (rotr(x,17)^rotr(x,19)^(x>>10U))

#define a Vals[0]
#define b Vals[1]
#define c Vals[2]
#define d Vals[3]
#define e Vals[4]
#define f Vals[5]
#define g Vals[6]
#define h Vals[7]

#define A Vals[0]
#define B Vals[1]
#define C Vals[2]
#define D Vals[3]
#define E Vals[4]
#define F Vals[5]
#define G Vals[6]
#define H Vals[7]

__kernel 
__attribute__((vec_type_hint(u)))
__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
void search(
   const uint state0, //a
   const uint state1, //b
   const uint state2, //c
   const uint state3, //b
   const uint state4, //e
   const uint state5, //f
   const uint state6, //g
   const uint state7, //h

   // second message block - the nonce is here
   const uint merkend,
   const uint time,  
   const uint target,
   const uint metro0,
   const uint metro1,
   const uint metro2,
   const uint metro3,
   const uint metro4,
   const uint metro5,
   const uint metro6,
   const uint metro7,

   // after this, 256 bits of 0
   // 32 bits of 80000000
   // 32 bits of 0
   // 32 bits of 000005A8
#ifndef GOFFSET
   const u base,
#endif
   volatile __global uint * output)

{

#ifdef GOFFSET
	const u nonce = (uint)(get_global_id(0));
#else
	const u nonce = base + (uint)(get_global_id(0));
#endif


#define  SHR(x,n) ((x & 0xFFFFFFFF) >> n)
#define ROTR(x,n) (SHR(x,n) | (x << (32 - n)))

#define S0(x) (ROTR(x, 7) ^ ROTR(x,18) ^  SHR(x, 3))
#define S1(x) (ROTR(x,17) ^ ROTR(x,19) ^  SHR(x,10))

#define S2(x) (ROTR(x, 2) ^ ROTR(x,13) ^ ROTR(x,22))
#define S3(x) (ROTR(x, 6) ^ ROTR(x,11) ^ ROTR(x,25))

#define F0(x,y,z) ((x & y) | (z & (x | y)))
#define F1(x,y,z) (z ^ (x & (y ^ z)))

#define R(t)                                    \
(                                               \
    W[t] = S1(W[t -  2]) + W[t -  7] +          \
           S0(W[t - 15]) + W[t - 16]            \
)

#define P(a,b,c,d,e,f,g,h,x,K)                  \
{                                               \
    temp1 = h + S3(e) + F1(e,f,g) + K + x;      \
    temp2 = S2(a) + F0(a,b,c);                  \
    d += temp1; h = temp1 + temp2;              \
}


    u Vals[8];
    u Last[8];
    u W[64];
    u t1=0;
    u t2=0;
    u temp1, temp2;

    a=state0;
    b=state1;
    c=state2;
    d=state3;
    e=state4;
    f=state5;
    g=state6;
    h=state7;


    W[0]=merkend;
    W[1]=metro0;
    W[2]=metro1;
    W[3]=metro2;
    W[4]=metro3;
    W[5]=metro4;
    W[6]=metro5;
    W[7]=metro6;
    W[8]=metro7;
    W[9]=time;
    W[10]=target;
    W[11]=nonce;
    W[12]=0x80000000;
    W[13]=0x00000000;
    W[14]=0x00000000;
    W[15]=0x00000380;

   t1 = h + E1(e) + ch(e,f,g) + K[0] + W[0];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[1] + W[1];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[2] + W[2];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[3] + W[3];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[4] + W[4];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[5] + W[5];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[6] + W[6];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[7] + W[7];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[8] + W[8];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[9] + W[9];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[10] + W[10];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[11] + W[11];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[12] + W[12];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[13] + W[13];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[14] + W[14];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[15] + W[15];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[16] = O1(W[14]) + W[9] + O0(W[1]) + W[0];t1 = h + E1(e) + ch(e,f,g) + K[16] + W[16];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[17] = O1(W[15]) + W[10] + O0(W[2]) + W[1];t1 = h + E1(e) + ch(e,f,g) + K[17] + W[17];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[18] = O1(W[16]) + W[11] + O0(W[3]) + W[2];t1 = h + E1(e) + ch(e,f,g) + K[18] + W[18];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[19] = O1(W[17]) + W[12] + O0(W[4]) + W[3];t1 = h + E1(e) + ch(e,f,g) + K[19] + W[19];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[20] = O1(W[18]) + W[13] + O0(W[5]) + W[4];t1 = h + E1(e) + ch(e,f,g) + K[20] + W[20];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[21] = O1(W[19]) + W[14] + O0(W[6]) + W[5];t1 = h + E1(e) + ch(e,f,g) + K[21] + W[21];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[22] = O1(W[20]) + W[15] + O0(W[7]) + W[6];t1 = h + E1(e) + ch(e,f,g) + K[22] + W[22];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[23] = O1(W[21]) + W[16] + O0(W[8]) + W[7];t1 = h + E1(e) + ch(e,f,g) + K[23] + W[23];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[24] = O1(W[22]) + W[17] + O0(W[9]) + W[8];t1 = h + E1(e) + ch(e,f,g) + K[24] + W[24];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[25] = O1(W[23]) + W[18] + O0(W[10]) + W[9];t1 = h + E1(e) + ch(e,f,g) + K[25] + W[25];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[26] = O1(W[24]) + W[19] + O0(W[11]) + W[10];t1 = h + E1(e) + ch(e,f,g) + K[26] + W[26];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[27] = O1(W[25]) + W[20] + O0(W[12]) + W[11];t1 = h + E1(e) + ch(e,f,g) + K[27] + W[27];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[28] = O1(W[26]) + W[21] + O0(W[13]) + W[12];t1 = h + E1(e) + ch(e,f,g) + K[28] + W[28];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[29] = O1(W[27]) + W[22] + O0(W[14]) + W[13];t1 = h + E1(e) + ch(e,f,g) + K[29] + W[29];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[30] = O1(W[28]) + W[23] + O0(W[15]) + W[14];t1 = h + E1(e) + ch(e,f,g) + K[30] + W[30];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[31] = O1(W[29]) + W[24] + O0(W[16]) + W[15];t1 = h + E1(e) + ch(e,f,g) + K[31] + W[31];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[32] = O1(W[30]) + W[25] + O0(W[17]) + W[16];t1 = h + E1(e) + ch(e,f,g) + K[32] + W[32];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[33] = O1(W[31]) + W[26] + O0(W[18]) + W[17];t1 = h + E1(e) + ch(e,f,g) + K[33] + W[33];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[34] = O1(W[32]) + W[27] + O0(W[19]) + W[18];t1 = h + E1(e) + ch(e,f,g) + K[34] + W[34];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[35] = O1(W[33]) + W[28] + O0(W[20]) + W[19];t1 = h + E1(e) + ch(e,f,g) + K[35] + W[35];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[36] = O1(W[34]) + W[29] + O0(W[21]) + W[20];t1 = h + E1(e) + ch(e,f,g) + K[36] + W[36];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[37] = O1(W[35]) + W[30] + O0(W[22]) + W[21];t1 = h + E1(e) + ch(e,f,g) + K[37] + W[37];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[38] = O1(W[36]) + W[31] + O0(W[23]) + W[22];t1 = h + E1(e) + ch(e,f,g) + K[38] + W[38];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[39] = O1(W[37]) + W[32] + O0(W[24]) + W[23];t1 = h + E1(e) + ch(e,f,g) + K[39] + W[39];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[40] = O1(W[38]) + W[33] + O0(W[25]) + W[24];t1 = h + E1(e) + ch(e,f,g) + K[40] + W[40];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[41] = O1(W[39]) + W[34] + O0(W[26]) + W[25];t1 = h + E1(e) + ch(e,f,g) + K[41] + W[41];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[42] = O1(W[40]) + W[35] + O0(W[27]) + W[26];t1 = h + E1(e) + ch(e,f,g) + K[42] + W[42];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[43] = O1(W[41]) + W[36] + O0(W[28]) + W[27];t1 = h + E1(e) + ch(e,f,g) + K[43] + W[43];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[44] = O1(W[42]) + W[37] + O0(W[29]) + W[28];t1 = h + E1(e) + ch(e,f,g) + K[44] + W[44];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[45] = O1(W[43]) + W[38] + O0(W[30]) + W[29];t1 = h + E1(e) + ch(e,f,g) + K[45] + W[45];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[46] = O1(W[44]) + W[39] + O0(W[31]) + W[30];t1 = h + E1(e) + ch(e,f,g) + K[46] + W[46];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[47] = O1(W[45]) + W[40] + O0(W[32]) + W[31];t1 = h + E1(e) + ch(e,f,g) + K[47] + W[47];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[48] = O1(W[46]) + W[41] + O0(W[33]) + W[32];t1 = h + E1(e) + ch(e,f,g) + K[48] + W[48];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[49] = O1(W[47]) + W[42] + O0(W[34]) + W[33];t1 = h + E1(e) + ch(e,f,g) + K[49] + W[49];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[50] = O1(W[48]) + W[43] + O0(W[35]) + W[34];t1 = h + E1(e) + ch(e,f,g) + K[50] + W[50];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[51] = O1(W[49]) + W[44] + O0(W[36]) + W[35];t1 = h + E1(e) + ch(e,f,g) + K[51] + W[51];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[52] = O1(W[50]) + W[45] + O0(W[37]) + W[36];t1 = h + E1(e) + ch(e,f,g) + K[52] + W[52];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[53] = O1(W[51]) + W[46] + O0(W[38]) + W[37];t1 = h + E1(e) + ch(e,f,g) + K[53] + W[53];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[54] = O1(W[52]) + W[47] + O0(W[39]) + W[38];t1 = h + E1(e) + ch(e,f,g) + K[54] + W[54];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[55] = O1(W[53]) + W[48] + O0(W[40]) + W[39];t1 = h + E1(e) + ch(e,f,g) + K[55] + W[55];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[56] = O1(W[54]) + W[49] + O0(W[41]) + W[40];t1 = h + E1(e) + ch(e,f,g) + K[56] + W[56];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[57] = O1(W[55]) + W[50] + O0(W[42]) + W[41];t1 = h + E1(e) + ch(e,f,g) + K[57] + W[57];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[58] = O1(W[56]) + W[51] + O0(W[43]) + W[42];t1 = h + E1(e) + ch(e,f,g) + K[58] + W[58];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[59] = O1(W[57]) + W[52] + O0(W[44]) + W[43];t1 = h + E1(e) + ch(e,f,g) + K[59] + W[59];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[60] = O1(W[58]) + W[53] + O0(W[45]) + W[44];t1 = h + E1(e) + ch(e,f,g) + K[60] + W[60];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[61] = O1(W[59]) + W[54] + O0(W[46]) + W[45];t1 = h + E1(e) + ch(e,f,g) + K[61] + W[61];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[62] = O1(W[60]) + W[55] + O0(W[47]) + W[46];t1 = h + E1(e) + ch(e,f,g) + K[62] + W[62];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[63] = O1(W[61]) + W[56] + O0(W[48]) + W[47];t1 = h + E1(e) + ch(e,f,g) + K[63] + W[63];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;


    W[0]=a+=state0;
    W[1]=b+=state1;
    W[2]=c+=state2;
    W[3]=d+=state3;
    W[4]=e+=state4;
    W[5]=f+=state5;
    W[6]=g+=state6;
    W[7]=h+=state7;
    W[8]=0x80000000;
    W[9]=0x00000000;
    W[10]=0x00000000;
    W[11]=0x00000000;
    W[12]=0x00000000;
    W[13]=0x00000000;
    W[14]=0x00000000;
    W[15]=0x00000100;

    a=initH[0];
    b=initH[1];
    c=initH[2];
    d=initH[3];
    e=initH[4];
    f=initH[5];
    g=initH[6];
    h=initH[7];

   t1 = h + E1(e) + ch(e,f,g) + K[0] + W[0];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[1] + W[1];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[2] + W[2];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[3] + W[3];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[4] + W[4];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[5] + W[5];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[6] + W[6];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[7] + W[7];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[8] + W[8];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[9] + W[9];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[10] + W[10];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[11] + W[11];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[12] + W[12];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[13] + W[13];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[14] + W[14];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
t1 = h + E1(e) + ch(e,f,g) + K[15] + W[15];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[16] = O1(W[14]) + W[9] + O0(W[1]) + W[0];t1 = h + E1(e) + ch(e,f,g) + K[16] + W[16];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[17] = O1(W[15]) + W[10] + O0(W[2]) + W[1];t1 = h + E1(e) + ch(e,f,g) + K[17] + W[17];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[18] = O1(W[16]) + W[11] + O0(W[3]) + W[2];t1 = h + E1(e) + ch(e,f,g) + K[18] + W[18];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[19] = O1(W[17]) + W[12] + O0(W[4]) + W[3];t1 = h + E1(e) + ch(e,f,g) + K[19] + W[19];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[20] = O1(W[18]) + W[13] + O0(W[5]) + W[4];t1 = h + E1(e) + ch(e,f,g) + K[20] + W[20];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[21] = O1(W[19]) + W[14] + O0(W[6]) + W[5];t1 = h + E1(e) + ch(e,f,g) + K[21] + W[21];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[22] = O1(W[20]) + W[15] + O0(W[7]) + W[6];t1 = h + E1(e) + ch(e,f,g) + K[22] + W[22];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[23] = O1(W[21]) + W[16] + O0(W[8]) + W[7];t1 = h + E1(e) + ch(e,f,g) + K[23] + W[23];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[24] = O1(W[22]) + W[17] + O0(W[9]) + W[8];t1 = h + E1(e) + ch(e,f,g) + K[24] + W[24];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[25] = O1(W[23]) + W[18] + O0(W[10]) + W[9];t1 = h + E1(e) + ch(e,f,g) + K[25] + W[25];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[26] = O1(W[24]) + W[19] + O0(W[11]) + W[10];t1 = h + E1(e) + ch(e,f,g) + K[26] + W[26];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[27] = O1(W[25]) + W[20] + O0(W[12]) + W[11];t1 = h + E1(e) + ch(e,f,g) + K[27] + W[27];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[28] = O1(W[26]) + W[21] + O0(W[13]) + W[12];t1 = h + E1(e) + ch(e,f,g) + K[28] + W[28];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[29] = O1(W[27]) + W[22] + O0(W[14]) + W[13];t1 = h + E1(e) + ch(e,f,g) + K[29] + W[29];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[30] = O1(W[28]) + W[23] + O0(W[15]) + W[14];t1 = h + E1(e) + ch(e,f,g) + K[30] + W[30];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[31] = O1(W[29]) + W[24] + O0(W[16]) + W[15];t1 = h + E1(e) + ch(e,f,g) + K[31] + W[31];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[32] = O1(W[30]) + W[25] + O0(W[17]) + W[16];t1 = h + E1(e) + ch(e,f,g) + K[32] + W[32];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[33] = O1(W[31]) + W[26] + O0(W[18]) + W[17];t1 = h + E1(e) + ch(e,f,g) + K[33] + W[33];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[34] = O1(W[32]) + W[27] + O0(W[19]) + W[18];t1 = h + E1(e) + ch(e,f,g) + K[34] + W[34];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[35] = O1(W[33]) + W[28] + O0(W[20]) + W[19];t1 = h + E1(e) + ch(e,f,g) + K[35] + W[35];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[36] = O1(W[34]) + W[29] + O0(W[21]) + W[20];t1 = h + E1(e) + ch(e,f,g) + K[36] + W[36];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[37] = O1(W[35]) + W[30] + O0(W[22]) + W[21];t1 = h + E1(e) + ch(e,f,g) + K[37] + W[37];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[38] = O1(W[36]) + W[31] + O0(W[23]) + W[22];t1 = h + E1(e) + ch(e,f,g) + K[38] + W[38];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[39] = O1(W[37]) + W[32] + O0(W[24]) + W[23];t1 = h + E1(e) + ch(e,f,g) + K[39] + W[39];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[40] = O1(W[38]) + W[33] + O0(W[25]) + W[24];t1 = h + E1(e) + ch(e,f,g) + K[40] + W[40];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[41] = O1(W[39]) + W[34] + O0(W[26]) + W[25];t1 = h + E1(e) + ch(e,f,g) + K[41] + W[41];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[42] = O1(W[40]) + W[35] + O0(W[27]) + W[26];t1 = h + E1(e) + ch(e,f,g) + K[42] + W[42];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[43] = O1(W[41]) + W[36] + O0(W[28]) + W[27];t1 = h + E1(e) + ch(e,f,g) + K[43] + W[43];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[44] = O1(W[42]) + W[37] + O0(W[29]) + W[28];t1 = h + E1(e) + ch(e,f,g) + K[44] + W[44];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[45] = O1(W[43]) + W[38] + O0(W[30]) + W[29];t1 = h + E1(e) + ch(e,f,g) + K[45] + W[45];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[46] = O1(W[44]) + W[39] + O0(W[31]) + W[30];t1 = h + E1(e) + ch(e,f,g) + K[46] + W[46];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[47] = O1(W[45]) + W[40] + O0(W[32]) + W[31];t1 = h + E1(e) + ch(e,f,g) + K[47] + W[47];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[48] = O1(W[46]) + W[41] + O0(W[33]) + W[32];t1 = h + E1(e) + ch(e,f,g) + K[48] + W[48];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[49] = O1(W[47]) + W[42] + O0(W[34]) + W[33];t1 = h + E1(e) + ch(e,f,g) + K[49] + W[49];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[50] = O1(W[48]) + W[43] + O0(W[35]) + W[34];t1 = h + E1(e) + ch(e,f,g) + K[50] + W[50];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[51] = O1(W[49]) + W[44] + O0(W[36]) + W[35];t1 = h + E1(e) + ch(e,f,g) + K[51] + W[51];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[52] = O1(W[50]) + W[45] + O0(W[37]) + W[36];t1 = h + E1(e) + ch(e,f,g) + K[52] + W[52];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[53] = O1(W[51]) + W[46] + O0(W[38]) + W[37];t1 = h + E1(e) + ch(e,f,g) + K[53] + W[53];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[54] = O1(W[52]) + W[47] + O0(W[39]) + W[38];t1 = h + E1(e) + ch(e,f,g) + K[54] + W[54];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[55] = O1(W[53]) + W[48] + O0(W[40]) + W[39];t1 = h + E1(e) + ch(e,f,g) + K[55] + W[55];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[56] = O1(W[54]) + W[49] + O0(W[41]) + W[40];t1 = h + E1(e) + ch(e,f,g) + K[56] + W[56];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[57] = O1(W[55]) + W[50] + O0(W[42]) + W[41];t1 = h + E1(e) + ch(e,f,g) + K[57] + W[57];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[58] = O1(W[56]) + W[51] + O0(W[43]) + W[42];t1 = h + E1(e) + ch(e,f,g) + K[58] + W[58];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[59] = O1(W[57]) + W[52] + O0(W[44]) + W[43];t1 = h + E1(e) + ch(e,f,g) + K[59] + W[59];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[60] = O1(W[58]) + W[53] + O0(W[45]) + W[44];t1 = h + E1(e) + ch(e,f,g) + K[60] + W[60];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[61] = O1(W[59]) + W[54] + O0(W[46]) + W[45];t1 = h + E1(e) + ch(e,f,g) + K[61] + W[61];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[62] = O1(W[60]) + W[55] + O0(W[47]) + W[46];t1 = h + E1(e) + ch(e,f,g) + K[62] + W[62];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
W[63] = O1(W[61]) + W[56] + O0(W[48]) + W[47];t1 = h + E1(e) + ch(e,f,g) + K[63] + W[63];t2 = E0(a) + Ma(a,b,c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;

	
    //a+=initH[0];
    //b+=initH[1];
    //c+=initH[2];
    //d+=initH[3];
    //e+=initH[4];
    //f+=initH[5];
    g+=initH[6];
    h+=initH[7];


	//printf("input %08x %08x %08x %08x %08x %08x %08x %08x \n",a,b,c,d,e,f,g,h);

    #define FOUND (0x0F)
    #define SETFOUND(Xnonce) output[output[FOUND]++] = Xnonce

    #if defined(VECTORS2)||defined(VECTORS4)
    if (any(h==0)) { // 32 zeros at least
    	if (h.x==0)
		SETFOUND(nonce.x);
	if (h.y==0)
		SETFOUND(nonce.y);
    #if defined(VECTORS4)
    	if (h.z==0)
		SETFOUND(nonce.z);
	if(h.w ==0)
		SETFOUND(nonce.w);

    #endif
    }

    #else
        if (h==0) { // 32 zeros at least
        SETFOUND(nonce);
    }

    #endif

}
