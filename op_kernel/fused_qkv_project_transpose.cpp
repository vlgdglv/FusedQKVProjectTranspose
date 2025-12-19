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
    }

    __aicore__ inline void Process() {
        uint32_t task_id = GetBlockIdx();
        uint32_t head_id = task_id % nH; 
        // PRINTF("[Kernel] blockIdx(Task id): %d doing shit\n", task_id);

        if (task_id >= nH) return;  // Do absloutly nothing

        uint64_t w_stride = (uint64_t)D * (uint64_t)Dh;

        // Calc Q
        mm.SetTensorA(this->hiddenGm);
        mm.SetTensorB(this->wqGm[w_stride * head_id]); 
        mm.DisableBias();
        mm.IterateAll(this->qGm[this->Dh * head_id]);
        mm.End();
        
        // Calc K
        mm.SetTensorA(this->hiddenGm);
        mm.SetTensorB(this->wkGm[w_stride * head_id]); 
        mm.DisableBias();
        mm.IterateAll(this->kGm[this->Dh * head_id]);
        mm.End();
        
        // Calc V
        mm.SetTensorA(this->hiddenGm);
        mm.SetTensorB(this->wvGm[w_stride * head_id]); 
        mm.DisableBias();
        mm.IterateAll(this->vGm[this->Dh * head_id]);
        mm.End();

        // PRINTF("[Kernel] I processd head No.%d.\n", head_id);
    }


public:
    using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using bType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using cType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
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
