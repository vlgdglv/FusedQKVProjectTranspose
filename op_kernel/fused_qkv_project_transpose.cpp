#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;


class KernelFusedQKVProjectTranspose {
public:
    __aicore__ inline KernelFusedQKVProjectTranspose() {}

    __aicore__ inline void Init(GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace,
        const TilingData& tilingData) {
        
        this->tilingData = tilingData;
        this->B = tilingData.batch;
        this->S = tilingData.seq_len;
        this->D = tilingData.hidden;
        this->nH = tilingData.num_heads;
        this->nHkv = tilingData.num_kv_heads;
        this->Dh = tilingData.head_dim;
        this->kvDh = tilingData.head_kv_dim;

        // PRINTF("[Kernel] Init begin, B: %d, S: %d, D: %d, nH: %d, Dh: %d.\n", B, S, D, nH, Dh);
        
        /*
            hiddenGm shape: [B, S, D]
            wXGm shape: [nH, D, Dh] should be ensured by caller
        */
        this->hiddenGm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden_states);
        this->wqGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_q);
        this->wkGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_k);
        this->wvGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_v);
        this->bqGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_q);
        this->bkGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_k);
        this->bvGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_v);
        this->qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        this->kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
        this->vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v);

        // pipe.InitBuffer(aBuf, D * sizeof(bfloat16_t));
        pipe.InitBuffer(cBuf, 3 * Dh * sizeof(bfloat16_t));

        // uint64_t sysWsBytes = tilingData.sys_workspace_size;
        // __gm__ char* base = (__gm__ char*)workspace;
        // __gm__ bfloat16_t* user = (__gm__ bfloat16_t*)(base + sysWsBytes);

        // workGm.SetGlobalBuffer(user, (uint64_t)nH * 3ull * (uint64_t)Dh);
    }

    __aicore__ inline void Process() {
        uint32_t head_id = GetBlockIdx();
        if (head_id >= nH) return;

        // PRINTF("[Kernel] blockIdx(Task id): %d doing shit\n", task_id);
        uint64_t w_stride = (uint64_t)D * (uint64_t)Dh * 3;
        // uint64_t c_offset = (uint64_t)head_id * (uint64_t)(3u * Dh);
        LocalTensor<bfloat16_t> c = cBuf.AllocTensor<bfloat16_t>();
        
        mm.SetTensorA(this->hiddenGm);
        mm.SetTensorB(this->wqGm[w_stride * head_id]); 
        mm.DisableBias();
        mm.IterateAll(c);
        mm.End();

        // DataCopy(tmp, this->workGm[c_offset], 3 * Dh);
        pipe_barrier(PIPE_ALL);
        DataCopy(this->qGm[Dh * head_id], c[0],        Dh);
        DataCopy(this->kGm[Dh * head_id], c[Dh],       Dh);
        DataCopy(this->vGm[Dh * head_id], c[2 * Dh],   Dh);
        pipe_barrier(PIPE_ALL);
        
        cBuf.FreeTensor(c);
        // tmpBuf.FreeTensor(tmp);
    }


public:
    // using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using bType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using cType     = MatmulType<TPosition::VECIN, CubeFormat::ND, bfloat16_t>;
    using biasType  = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<aType, bType, cType, biasType> mm;
    // Matmul<aType, bType, cType, biasType> mmQ;
    // Matmul<aType, bType, cType, biasType> mmK;
    // Matmul<aType, bType, cType, biasType> mmV;
    TPipe pipe;

private:
    TilingData tilingData;

    GlobalTensor<bfloat16_t> hiddenGm, wqGm, wkGm, wvGm;
    GlobalTensor<bfloat16_t> bqGm, bkGm, bvGm;
    GlobalTensor<bfloat16_t> qGm, kGm, vGm;
    
    uint32_t B, S, D, nH, nHkv, Dh, kvDh;

    // TBuf<TPosition::VECOUT> aBuf;
    TBuf<TPosition::VECIN> tmpBuf;
    TBuf<TPosition::VECIN> cBuf;
    
    GlobalTensor<bfloat16_t> workGm;

};

extern "C" __global__ __aicore__ 
    void fused_qkv_project_transpose(
        GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedQKVProjectTranspose op;
    
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tiling_data.cube_tiling);
    op.Init(hidden_states, w_q, w_k, w_v, b_q, b_k, b_v, q, k, v, workspace, tiling_data);
    op.Process();

    // REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(),
    //               op.mmQ, &tiling_data.cube_tiling,
    //               op.mmK, &tiling_data.cube_tiling,
    //               op.mmV, &tiling_data.cube_tiling);

}
