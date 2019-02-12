/*
 * Copyright 2011-2013 Con Kolivas
 * Copyright 2012-2013 Luke Dashjr
 * Copyright 2011 Nils Schneider
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <pthread.h>
#include <string.h>

#include "findnonce.h"
#include "miner.h"

#ifdef USE_SHA256D
const uint32_t SHA256_K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define rotate(x,y) ((x<<y) | (x>>(sizeof(x)*8-y)))
#define rotr(x,y) ((x>>y) | (x<<(sizeof(x)*8-y)))

#define R(a, b, c, d, e, f, g, h, w, k) \
	h = h + (rotate(e, 26) ^ rotate(e, 21) ^ rotate(e, 7)) + (g ^ (e & (f ^ g))) + k + w; \
	d = d + h; \
	h = h + (rotate(a, 30) ^ rotate(a, 19) ^ rotate(a, 10)) + ((a & b) | (c & (a | b)))

void precalc_hash(struct opencl_work_data *blk, uint32_t *state, uint32_t *data)
{

    blk->state0=state[0];
    blk->state1=state[1];
    blk->state2=state[2];
    blk->state3=state[3];
    blk->state4=state[4];
    blk->state5=state[5];
    blk->state6=state[6];
    blk->state7=state[7];
    blk->merkend=data[0];
    blk->metro0=data[1];
    blk->metro1=data[2];
    blk->metro2=data[3];
    blk->metro3=data[4];
    blk->metro4=data[5];
    blk->metro5=data[6];
    blk->metro6=data[7];
    blk->metro7=data[8];
    blk->time=data[9];
    blk->target=data[10];

}
#endif

struct pc_data {
	struct thr_info *thr;
	struct work work;
	uint32_t res[OPENCL_MAX_BUFFERSIZE];
	pthread_t pth;
	int found;
	enum cl_kernels kinterface;
};

static void *postcalc_hash(void *userdata)
{
	struct pc_data *pcd = (struct pc_data *)userdata;
	struct thr_info *thr = pcd->thr;
	unsigned int entry = 0;
	int found = FOUND;
#ifdef USE_SCRYPT
	if (pcd->kinterface == KL_SCRYPT)
		found = SCRYPT_FOUND;
#endif

	pthread_detach(pthread_self());
	RenameThread("postcalchsh");

	/* To prevent corrupt values in FOUND from trying to read beyond the
	 * end of the res[] array */
	if (unlikely(pcd->res[found] & ~found)) {
		applog(LOG_WARNING, "%"PRIpreprv": invalid nonce count - HW error",
				thr->cgpu->proc_repr);
		inc_hw_errors_only(thr);
		pcd->res[found] &= found;
	}

	for (entry = 0; entry < pcd->res[found]; entry++) {
		uint32_t nonce = pcd->res[entry];
#ifdef USE_OPENCL_FULLHEADER
		if (pcd->kinterface == KL_FULLHEADER)
			nonce = swab32(nonce);
#endif

		applog(LOG_DEBUG, "OCL NONCE %u found in slot %d", nonce, entry);
		submit_nonce(thr, &pcd->work, nonce);
	}

	clean_work(&pcd->work);
	free(pcd);

	return NULL;
}

void postcalc_hash_async(struct thr_info * const thr, struct work * const work, uint32_t * const res, const enum cl_kernels kinterface)
{
	struct pc_data *pcd = malloc(sizeof(struct pc_data));
	int buffersize;

	if (unlikely(!pcd)) {
		applog(LOG_ERR, "Failed to malloc pc_data in postcalc_hash_async");
		return;
	}

	*pcd = (struct pc_data){
		.thr = thr,
		.kinterface = kinterface,
	};
	__copy_work(&pcd->work, work);
#ifdef USE_SCRYPT
	if (kinterface == KL_SCRYPT)
		buffersize = SCRYPT_BUFFERSIZE;
	else
#endif
		buffersize = BUFFERSIZE;
	memcpy(&pcd->res, res, buffersize);

	if (pthread_create(&pcd->pth, NULL, postcalc_hash, (void *)pcd)) {
		applog(LOG_ERR, "Failed to create postcalc_hash thread");
		return;
	}
}
